from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Instruction:
    """Representa uma instrução MIPS simplificada usada pelo simulador."""

    id: int
    text: str
    op: str
    rd: Optional[int] = None
    rs: Optional[int] = None
    rt: Optional[int] = None
    imm: Optional[int] = None
    label: Optional[str] = None
    target_pc: Optional[int] = None

    issue_cycle: Optional[int] = None
    start_exec_cycle: Optional[int] = None
    end_exec_cycle: Optional[int] = None
    write_result_cycle: Optional[int] = None
    commit_cycle: Optional[int] = None

    rob_index: Optional[int] = None
    seq_num: Optional[int] = None  # ordem global de despacho


@dataclass
class ReservationStationEntry:
    """Entrada de estação de reserva (Reservation Station)."""

    name: str
    kind: str  # "ADD", "MUL", "LS"
    busy: bool = False
    op: Optional[str] = None
    Vj: Optional[int] = None
    Vk: Optional[int] = None
    Qj: Optional[int] = None  # índice no ROB que produzirá Vj
    Qk: Optional[int] = None  # índice no ROB que produzirá Vk
    dest_rob: Optional[int] = None
    remaining_cycles: int = 0
    instr_id: Optional[int] = None


@dataclass
class ROBEntry:
    """Entrada do buffer de reordenação (Reorder Buffer)."""

    index: int
    busy: bool = False
    instr_id: Optional[int] = None
    seq_num: Optional[int] = None

    dest_reg: Optional[int] = None
    value: Optional[int] = None
    ready: bool = False
    state: str = "ISSUE"  # ISSUE, EXEC, WRITE, COMMIT, FLUSHED

    is_store: bool = False
    store_addr: Optional[int] = None
    store_value: Optional[int] = None

    is_branch: bool = False
    branch_target: Optional[int] = None
    branch_pred_taken: bool = False
    branch_actual_taken: Optional[bool] = None
    rat_snapshot: Optional[Dict[int, Optional[int]]] = None


class TomasuloSimulator:
    """
    Simulador didático do algoritmo de Tomasulo com ROB e especulação de desvio.

    - Instruções MIPS simplificadas
    - Renomeação de registradores (RAT)
    - Estações de reserva por tipo de funcionalidade
    - Buffer de reordenação (ROB)
    - Execução especulativa de desvios (previsão estática: não tomado)
    """

    def __init__(
        self,
        program: List[Instruction],
        rob_size: int = 16,
        num_add_rs: int = 4,
        num_mul_rs: int = 2,
        num_ls_rs: int = 3,
        issue_width: int = 2,
        commit_width: int = 2,
    ) -> None:
        self.program: List[Instruction] = program
        self.rob_size = rob_size
        self.issue_width = issue_width
        self.commit_width = commit_width

        # Estado arquitetural
        self.registers: List[int] = [0] * 32
        self.memory: Dict[int, int] = {}

        # RAT: mapeia registrador lógico -> entrada do ROB (índice)
        self.rat: Dict[int, Optional[int]] = {i: None for i in range(32)}

        # ROB como buffer circular
        self.rob: List[ROBEntry] = [ROBEntry(index=i) for i in range(self.rob_size)]
        self.rob_head: int = 0
        self.rob_tail: int = 0
        self.rob_count: int = 0

        # Estações de reserva
        self.rs_add: List[ReservationStationEntry] = [
            ReservationStationEntry(name=f"Add{i}", kind="ADD")
            for i in range(num_add_rs)
        ]
        self.rs_mul: List[ReservationStationEntry] = [
            ReservationStationEntry(name=f"Mul{i}", kind="MUL")
            for i in range(num_mul_rs)
        ]
        self.rs_ls: List[ReservationStationEntry] = [
            ReservationStationEntry(name=f"LS{i}", kind="LS")
            for i in range(num_ls_rs)
        ]

        # Latências por operação (em ciclos)
        self.latency: Dict[str, int] = {
            "ADD": 1,
            "SUB": 1,
            "ADDI": 1,
            "MUL": 3,
            "DIV": 5,
            "LD": 2,
            "ST": 2,
            "BEQ": 1,
            "BNE": 1,
            "NOP": 1,
        }

        # Controle global da simulação
        self.cycle: int = 0
        self.pc: int = 0  # índice na lista de instruções
        self.seq_counter: int = 0

        # Métricas
        self.total_instructions: int = len(self.program)
        self.committed_instructions: int = 0
        self.bubble_cycles: int = 0

        self.finished: bool = False

        # Mapa rápido de id -> instrução
        self._instr_map: Dict[int, Instruction] = {ins.id: ins for ins in self.program}

    # ------------------------------------------------------------------
    # Resumo das métricas
    # ------------------------------------------------------------------
    @property
    def ipc(self) -> float:
        if self.cycle == 0:
            return 0.0
        return self.committed_instructions / float(self.cycle)

    # ------------------------------------------------------------------
    # Funções auxiliares
    # ------------------------------------------------------------------
    def reset_state(self) -> None:
        """Reseta o estado dinâmico, mantendo o mesmo programa."""
        self.registers = [0] * 32
        self.memory = {}
        self.rat = {i: None for i in range(32)}

        # Reset de ponteiros do ROB
        self.rob_head = 0
        self.rob_tail = 0
        self.rob_count = 0

        for entry in self.rob:
            entry.busy = False
            entry.instr_id = None
            entry.seq_num = None
            entry.dest_reg = None
            entry.value = None
            entry.ready = False
            entry.state = "ISSUE"
            entry.is_store = False
            entry.store_addr = None
            entry.store_value = None
            entry.is_branch = False
            entry.branch_target = None
            entry.branch_pred_taken = False
            entry.branch_actual_taken = None
            entry.rat_snapshot = None

        for rs in self.rs_add + self.rs_mul + self.rs_ls:
            rs.busy = False
            rs.op = None
            rs.Vj = None
            rs.Vk = None
            rs.Qj = None
            rs.Qk = None
            rs.dest_rob = None
            rs.remaining_cycles = 0
            rs.instr_id = None

        self.cycle = 0
        self.pc = 0
        self.seq_counter = 0
        self.committed_instructions = 0
        self.bubble_cycles = 0
        self.finished = False

        for ins in self.program:
            ins.issue_cycle = None
            ins.start_exec_cycle = None
            ins.end_exec_cycle = None
            ins.write_result_cycle = None
            ins.commit_cycle = None
            ins.rob_index = None
            ins.seq_num = None

    def _all_rs(self) -> List[ReservationStationEntry]:
        return self.rs_add + self.rs_mul + self.rs_ls

    def has_inflight_work(self) -> bool:
        """Retorna True se ainda há trabalho pendente na máquina."""
        if self.pc < len(self.program):
            return True
        if self.rob_count > 0:
            return True
        for rs in self._all_rs():
            if rs.busy:
                return True
        return False

    # ------------------------------------------------------------------
    # Passo principal de simulação: um ciclo
    # ------------------------------------------------------------------
    def step_cycle(self) -> None:
        """Simula um ciclo de clock."""
        if self.finished:
            return

        self.cycle += 1

        # 1) Avançar execução nas RS (inclui terminar execuções e broadcast)
        self._advance_execution()

        # 2) Commit no ROB
        committed_this_cycle = self._commit_from_rob()

        # 3) Despacho de novas instruções
        self._issue_instructions()

        # 4) Verificar se terminou
        if not self.has_inflight_work():
            self.finished = True

        # 5) Atualizar métricas de bolha
        if committed_this_cycle == 0 and self.has_inflight_work():
            self.bubble_cycles += 1

    # ------------------------------------------------------------------
    # Execução / broadcast
    # ------------------------------------------------------------------
    def _advance_execution(self) -> None:
        """Avança a execução nas estações de reserva."""
        # Primeiro: terminar quem estava com remaining_cycles == 1
        for rs in self._all_rs():
            if rs.busy and rs.remaining_cycles == 1:
                self._finish_execution(rs)

        # Segundo: decrementar os demais em execução
        for rs in self._all_rs():
            if rs.busy and rs.remaining_cycles > 1:
                rs.remaining_cycles -= 1

        # Terceiro: iniciar execução em RS prontas (operandos disponíveis)
        for rs in self._all_rs():
            if rs.busy and rs.remaining_cycles == 0 and rs.Qj is None and rs.Qk is None:
                op = rs.op
                if not op:
                    continue
                latency = self.latency.get(op, 1)
                rs.remaining_cycles = latency
                ins = self._instr_map.get(rs.instr_id) if rs.instr_id is not None else None
                if ins and ins.start_exec_cycle is None:
                    ins.start_exec_cycle = self.cycle

    def _finish_execution(self, rs: ReservationStationEntry) -> None:
        """Finaliza a execução de uma instrução em uma RS e faz broadcast."""
        ins = self._instr_map.get(rs.instr_id) if rs.instr_id is not None else None
        if ins is None or rs.dest_rob is None:
            # Pode ser um branch ou store que não escreve registrador diretamente
            rob_entry = self._get_rob_entry(rs.dest_rob) if rs.dest_rob is not None else None
        else:
            rob_entry = self._get_rob_entry(rs.dest_rob)

        if ins:
            ins.end_exec_cycle = self.cycle

        if rob_entry is None and ins is not None and ins.op in {"BEQ", "BNE"}:
            # Branch sem destino de registrador, mas ainda no ROB
            rob_entry = self._find_rob_by_instr(ins.id)

        if ins and rob_entry:
            if ins.op in {"ADD", "SUB", "MUL", "DIV", "ADDI"}:
                result = self._compute_alu_result(ins, rs)
                rob_entry.value = result
                rob_entry.ready = True
                rob_entry.state = "WRITE"
                if ins.write_result_cycle is None:
                    ins.write_result_cycle = self.cycle
                self._broadcast_result(rob_entry.index, result)
            elif ins.op == "LD":
                addr = self._compute_address(ins, rs)
                val = self.memory.get(addr, 0)
                rob_entry.value = val
                rob_entry.ready = True
                rob_entry.state = "WRITE"
                rob_entry.store_addr = addr  # apenas para visualização
                if ins.write_result_cycle is None:
                    ins.write_result_cycle = self.cycle
                self._broadcast_result(rob_entry.index, val)
            elif ins.op == "ST":
                addr = self._compute_address(ins, rs)
                value = rs.Vj if rs.Vj is not None else 0
                rob_entry.is_store = True
                rob_entry.store_addr = addr
                rob_entry.store_value = value
                rob_entry.ready = True
                rob_entry.state = "WRITE"
                if ins.write_result_cycle is None:
                    ins.write_result_cycle = self.cycle
            elif ins.op in {"BEQ", "BNE"}:
                taken = self._compute_branch_taken(ins, rs)
                rob_entry.is_branch = True
                rob_entry.branch_actual_taken = taken
                rob_entry.ready = True
                rob_entry.state = "WRITE"
                if ins.write_result_cycle is None:
                    ins.write_result_cycle = self.cycle
                # Verificar se houve erro de previsão (sempre previmos "não tomado")
                pred_taken = rob_entry.branch_pred_taken
                if taken != pred_taken:
                    # Misprediction -> flush especulativo
                    target_pc = ins.target_pc if taken else ins.id + 1
                    if target_pc is None:
                        target_pc = ins.id + 1
                    self._flush_after_mispredict(rob_entry, target_pc)

        # Libera RS
        rs.busy = False
        rs.op = None
        rs.Vj = None
        rs.Vk = None
        rs.Qj = None
        rs.Qk = None
        rs.dest_rob = None
        rs.remaining_cycles = 0
        rs.instr_id = None

    def _get_rob_entry(self, index: int) -> Optional[ROBEntry]:
        if 0 <= index < self.rob_size:
            return self.rob[index]
        return None

    def _find_rob_by_instr(self, instr_id: int) -> Optional[ROBEntry]:
        for entry in self.rob:
            if entry.busy and entry.instr_id == instr_id:
                return entry
        return None

    def _broadcast_result(self, rob_index: int, value: int) -> None:
        """Broadcast do resultado (CDB): atualiza RS que dependem deste ROB."""
        for rs in self._all_rs():
            if not rs.busy:
                continue
            if rs.Qj == rob_index:
                rs.Qj = None
                rs.Vj = value
            if rs.Qk == rob_index:
                rs.Qk = None
                rs.Vk = value

    # ------------------------------------------------------------------
    # Commit
    # ------------------------------------------------------------------
    def _commit_from_rob(self) -> int:
        committed = 0
        for _ in range(self.commit_width):
            if self.rob_count == 0:
                break
            entry = self.rob[self.rob_head]
            if not entry.busy:
                # Entrada vazia – avança ponteiro
                self.rob_head = (self.rob_head + 1) % self.rob_size
                self.rob_count -= 1
                continue

            if not entry.ready:
                # Não pode commit ainda
                break

            ins = self._instr_map.get(entry.instr_id) if entry.instr_id is not None else None

            # Commit de store
            if entry.is_store:
                if entry.store_addr is not None and entry.store_value is not None:
                    self.memory[entry.store_addr] = entry.store_value

            # Commit de branch: nada a escrever em registrador/memória
            if entry.is_branch:
                pass

            # Commit de instrução que escreve registrador
            if entry.dest_reg is not None and not entry.is_store and not entry.is_branch:
                if entry.dest_reg != 0:
                    self.registers[entry.dest_reg] = entry.value if entry.value is not None else 0
                # Atualiza RAT: se ainda apontar para esta entrada, limpa
                if self.rat.get(entry.dest_reg) == entry.index:
                    self.rat[entry.dest_reg] = None

            # Marcar instrução como commitada
            if ins:
                ins.commit_cycle = self.cycle

            entry.busy = False
            entry.ready = False
            entry.state = "COMMIT"
            entry.instr_id = None
            entry.dest_reg = None
            entry.value = None
            entry.is_store = False
            entry.store_addr = None
            entry.store_value = None
            entry.is_branch = False
            entry.branch_target = None
            entry.branch_pred_taken = False
            entry.branch_actual_taken = None
            entry.rat_snapshot = None

            self.rob_head = (self.rob_head + 1) % self.rob_size
            self.rob_count -= 1

            committed += 1
            self.committed_instructions += 1

        return committed

    # ------------------------------------------------------------------
    # Despacho (issue)
    # ------------------------------------------------------------------
    def _issue_instructions(self) -> None:
        issued = 0
        while issued < self.issue_width and self.pc < len(self.program):
            ins = self.program[self.pc]

            if ins.op == "NOP":
                # NOP entra no ROB apenas para manter ordem
                rob_index, _ = self._allocate_rob_entry(ins, dest_reg=None, is_branch=False)
                if rob_index is None:
                    # ROB cheio, não consegue issue
                    break
                ins.issue_cycle = self.cycle
                ins.rob_index = rob_index
                self.pc += 1
                issued += 1
                continue

            rs_pool = self._select_rs_pool(ins.op)
            if rs_pool is None:
                # Operação desconhecida
                break

            free_rs = next((rs for rs in rs_pool if not rs.busy), None)
            if free_rs is None:
                # Sem estação de reserva livre
                break

            # Verificar se há espaço no ROB
            dest_reg = self._dest_reg_for_instruction(ins)
            is_branch = ins.op in {"BEQ", "BNE"}
            rob_index, rob_entry = self._allocate_rob_entry(ins, dest_reg=dest_reg, is_branch=is_branch)
            if rob_index is None or rob_entry is None:
                # ROB cheio
                break

            # Preencher RS com operandos/depêndencias
            self._fill_rs_entry(free_rs, ins, rob_index)

            # Atualizar metadados da instrução
            if ins.issue_cycle is None:
                ins.issue_cycle = self.cycle
            ins.rob_index = rob_index
            ins.seq_num = rob_entry.seq_num

            # Atualizar PC (considerando branch especulativo ou normal)
            if ins.op in {"BEQ", "BNE"}:
                # Previsão estática: não tomado => PC segue sequencialmente
                rob_entry.is_branch = True
                rob_entry.branch_pred_taken = False
                rob_entry.branch_target = ins.target_pc
                # Snapshot do RAT para possível flush
                rob_entry.rat_snapshot = dict(self.rat)
                self.pc += 1
            else:
                self.pc += 1

            issued += 1

    def _allocate_rob_entry(
        self,
        ins: Instruction,
        dest_reg: Optional[int],
        is_branch: bool,
    ) -> Tuple[Optional[int], Optional[ROBEntry]]:
        if self.rob_count == self.rob_size:
            return None, None

        entry = self.rob[self.rob_tail]
        self.rob_tail = (self.rob_tail + 1) % self.rob_size
        self.rob_count += 1

        entry.busy = True
        entry.instr_id = ins.id
        entry.seq_num = self.seq_counter
        self.seq_counter += 1

        entry.dest_reg = dest_reg
        entry.value = None
        entry.ready = False
        entry.state = "ISSUE"
        entry.is_store = False
        entry.store_addr = None
        entry.store_value = None
        entry.is_branch = is_branch
        entry.branch_target = None
        entry.branch_pred_taken = False
        entry.branch_actual_taken = None
        entry.rat_snapshot = None

        # Atualizar RAT se tiver destino de registrador
        if dest_reg is not None and not is_branch:
            self.rat[dest_reg] = entry.index

        return entry.index, entry

    def _select_rs_pool(self, op: str) -> Optional[List[ReservationStationEntry]]:
        if op in {"ADD", "SUB", "ADDI", "BEQ", "BNE"}:
            return self.rs_add
        if op in {"MUL", "DIV"}:
            return self.rs_mul
        if op in {"LD", "ST"}:
            return self.rs_ls
        return None

    def _dest_reg_for_instruction(self, ins: Instruction) -> Optional[int]:
        if ins.op in {"ADD", "SUB", "MUL", "DIV", "ADDI", "LD"}:
            return ins.rd or ins.rt
        return None

    def _get_operand(self, reg: Optional[int]) -> Tuple[Optional[int], Optional[int]]:
        """Retorna (valor, dependenciaROB)."""
        if reg is None:
            return None, None
        rob_idx = self.rat.get(reg)
        if rob_idx is None:
            return self.registers[reg], None
        # Há dependência com entrada do ROB
        rob_entry = self._get_rob_entry(rob_idx)
        if rob_entry and rob_entry.ready:
            return rob_entry.value, None
        return None, rob_idx

    def _fill_rs_entry(self, rs: ReservationStationEntry, ins: Instruction, rob_index: int) -> None:
        rs.busy = True
        rs.op = ins.op
        rs.dest_rob = rob_index
        rs.remaining_cycles = 0
        rs.instr_id = ins.id

        if ins.op in {"ADD", "SUB", "MUL", "DIV"}:
            vj, qj = self._get_operand(ins.rs)
            vk, qk = self._get_operand(ins.rt)
            rs.Vj, rs.Qj = vj, qj
            rs.Vk, rs.Qk = vk, qk
        elif ins.op == "ADDI":
            vj, qj = self._get_operand(ins.rs)
            rs.Vj, rs.Qj = vj, qj
            rs.Vk, rs.Qk = ins.imm, None
        elif ins.op == "LD":
            # endereço = base + offset (imm)
            vj, qj = self._get_operand(ins.rs)
            rs.Vj, rs.Qj = vj, qj  # base
            rs.Vk, rs.Qk = ins.imm, None  # offset imediato
        elif ins.op == "ST":
            # store rt, offset(rs) -> valor em rt, endereço = rs + offset
            # Vj/Vk usados como (valor, base+offset)
            # Vj: valor a armazenar, Vk: base
            v_val, q_val = self._get_operand(ins.rt)
            v_base, q_base = self._get_operand(ins.rs)
            rs.Vj, rs.Qj = v_val, q_val
            rs.Vk, rs.Qk = v_base, q_base
        elif ins.op in {"BEQ", "BNE"}:
            vj, qj = self._get_operand(ins.rs)
            vk, qk = self._get_operand(ins.rt)
            rs.Vj, rs.Qj = vj, qj
            rs.Vk, rs.Qk = vk, qk

    # ------------------------------------------------------------------
    # Cálculo de resultados
    # ------------------------------------------------------------------
    def _compute_alu_result(self, ins: Instruction, rs: ReservationStationEntry) -> int:
        vj = rs.Vj or 0
        vk = rs.Vk or 0
        if ins.op == "ADD":
            return vj + vk
        if ins.op == "SUB":
            return vj - vk
        if ins.op == "MUL":
            return vj * vk
        if ins.op == "DIV":
            return vj // vk if vk != 0 else 0
        if ins.op == "ADDI":
            return vj + (ins.imm or 0)
        return 0

    def _compute_address(self, ins: Instruction, rs: ReservationStationEntry) -> int:
        # LD rt, offset(rs) ou ST rt, offset(rs)
        base = rs.Vk if ins.op == "ST" else rs.Vj
        base = base or 0
        offset = ins.imm or 0
        return base + offset

    def _compute_branch_taken(self, ins: Instruction, rs: ReservationStationEntry) -> bool:
        vj = rs.Vj or 0
        vk = rs.Vk or 0
        if ins.op == "BEQ":
            return vj == vk
        if ins.op == "BNE":
            return vj != vk
        return False

    # ------------------------------------------------------------------
    # Flush em caso de misprediction
    # ------------------------------------------------------------------
    def _flush_after_mispredict(self, branch_entry: ROBEntry, correct_pc: int) -> None:
        """Descarta instruções especulativas após um branch mal previsto."""
        branch_seq = branch_entry.seq_num
        # Limpa entradas do ROB mais novas que o branch
        removed = 0
        for entry in self.rob:
            if (
                entry.busy
                and entry.seq_num is not None
                and branch_seq is not None
                and entry.seq_num > branch_seq
            ):
                entry.busy = False
                entry.ready = False
                entry.state = "FLUSHED"
                entry.instr_id = None
                entry.dest_reg = None
                entry.value = None
                entry.is_store = False
                entry.store_addr = None
                entry.store_value = None
                entry.is_branch = False
                entry.branch_target = None
                entry.branch_pred_taken = False
                entry.branch_actual_taken = None
                entry.rat_snapshot = None
                removed += 1

        self.rob_count = max(0, self.rob_count - removed)

        # Limpa estações de reserva com instruções mais novas
        for rs in self._all_rs():
            if not rs.busy or rs.instr_id is None:
                continue
            ins = self._instr_map.get(rs.instr_id)
            if ins and ins.seq_num is not None and branch_seq is not None and ins.seq_num > branch_seq:
                rs.busy = False
                rs.op = None
                rs.Vj = None
                rs.Vk = None
                rs.Qj = None
                rs.Qk = None
                rs.dest_rob = None
                rs.remaining_cycles = 0
                rs.instr_id = None

        # Restaura RAT para o snapshot do branch
        if branch_entry.rat_snapshot is not None:
            self.rat = dict(branch_entry.rat_snapshot)

        # Ajusta PC para o caminho correto
        self.pc = correct_pc

    # ------------------------------------------------------------------
    # Funções de apoio à interface
    # ------------------------------------------------------------------
    def get_instruction_stage(self, ins: Instruction) -> str:
        if ins.commit_cycle is not None and ins.commit_cycle <= self.cycle:
            return "Commit"
        if ins.write_result_cycle is not None and ins.write_result_cycle <= self.cycle:
            return "WriteResult"
        if ins.start_exec_cycle is not None and (
            ins.end_exec_cycle is None or ins.end_exec_cycle >= self.cycle
        ):
            return "Execução"
        if ins.issue_cycle is not None and ins.issue_cycle <= self.cycle:
            return "Despacho"
        return "Não emitida"


# ----------------------------------------------------------------------
# Parser de assembly MIPS simplificado
# ----------------------------------------------------------------------

import re


def parse_register(token: str) -> int:
    token = token.strip().upper()
    if token.startswith("R"):
        return int(token[1:])
    return int(token)


def parse_offset_reg(token: str) -> Tuple[int, int]:
    """
    Converte "offset(Rx)" em (offset, reg).
    Ex: "4(R1)" -> (4, 1)
    """
    token = token.strip()
    m = re.match(r"(-?\d+)\((R?\d+)\)", token.replace(" ", ""))
    if not m:
        raise ValueError(f"Endereço inválido: {token}")
    offset = int(m.group(1))
    reg = parse_register(m.group(2))
    return offset, reg


def parse_assembly(asm: str) -> List[Instruction]:
    """
    Parser simples de assembly MIPS para o subconjunto usado no simulador.
    Suporta:
      - ADD rd, rs, rt
      - SUB rd, rs, rt
      - MUL rd, rs, rt
      - DIV rd, rs, rt
      - ADDI rd, rs, imm
      - LD rt, offset(rs)
      - ST rt, offset(rs)
      - BEQ rs, rt, label
      - BNE rs, rt, label
      - NOP
    """
    lines = asm.splitlines()
    cleaned: List[Tuple[str, int]] = []
    labels: Dict[str, int] = {}

    # Primeira passagem: limpar comentários, extrair labels
    pc = 0
    for idx, line in enumerate(lines):
        # Remove comentários
        line = re.split(r"[#;]", line, maxsplit=1)[0]
        line = line.strip()
        if not line:
            continue

        # Lida com labels do tipo "LOOP:" ou "LOOP: ADD R1, R2, R3"
        while ":" in line:
            label, rest = line.split(":", 1)
            label = label.strip()
            if label:
                labels[label] = pc
            line = rest.strip()
            if not line:
                break

        if not line:
            continue

        cleaned.append((line, idx + 1))
        pc += 1

    # Segunda passagem: criar instruções
    instructions: List[Instruction] = []
    for ins_id, (line, _lineno) in enumerate(cleaned):
        tokens = re.split(r"[,\s]+", line)
        tokens = [t for t in tokens if t]
        if not tokens:
            continue
        op = tokens[0].upper()
        args = tokens[1:]

        inst = Instruction(id=ins_id, text=line, op=op)

        if op in {"ADD", "SUB", "MUL", "DIV"}:
            if len(args) != 3:
                raise ValueError(f"Instrução {op} espera 3 operandos: {line}")
            rd = parse_register(args[0])
            rs = parse_register(args[1])
            rt = parse_register(args[2])
            inst.rd, inst.rs, inst.rt = rd, rs, rt
        elif op == "ADDI":
            if len(args) != 3:
                raise ValueError("ADDI espera 3 operandos: rd, rs, imm")
            rd = parse_register(args[0])
            rs = parse_register(args[1])
            imm = int(args[2])
            inst.rd, inst.rs, inst.imm = rd, rs, imm
        elif op == "LD":
            if len(args) != 2:
                raise ValueError("LD espera 2 operandos: rt, offset(rs)")
            rt = parse_register(args[0])
            offset, base = parse_offset_reg(args[1])
            inst.rt, inst.rs, inst.imm = rt, base, offset
        elif op == "ST":
            if len(args) != 2:
                raise ValueError("ST espera 2 operandos: rt, offset(rs)")
            rt = parse_register(args[0])
            offset, base = parse_offset_reg(args[1])
            inst.rt, inst.rs, inst.imm = rt, base, offset
        elif op in {"BEQ", "BNE"}:
            if len(args) != 3:
                raise ValueError(f"{op} espera 3 operandos: rs, rt, label")
            rs = parse_register(args[0])
            rt = parse_register(args[1])
            label = args[2]
            inst.rs, inst.rt, inst.label = rs, rt, label
        elif op == "NOP":
            # Nada a fazer
            pass
        else:
            raise ValueError(f"Operação não suportada: {op}")

        instructions.append(inst)

    # Resolver labels de branch
    for inst in instructions:
        if inst.op in {"BEQ", "BNE"} and inst.label:
            if inst.label not in labels:
                raise ValueError(f"Label não definida: {inst.label}")
            inst.target_pc = labels[inst.label]

    return instructions


# ----------------------------------------------------------------------
# Programas de exemplo
# ----------------------------------------------------------------------

SAMPLE_PROGRAM_1 = """\
# Exemplo 1: soma simples com desvio condicional
    LD  R1, 0(R0)      # R1 = MEM[0]
    LD  R2, 4(R0)      # R2 = MEM[4]
    ADD R3, R1, R2     # R3 = R1 + R2
    ST  R3, 8(R0)      # MEM[8] = R3
    BEQ R3, R0, FIM    # se resultado == 0, salta
    ADDI R4, R4, 1     # caso contrário, incrementa R4
FIM:
    NOP
"""

SAMPLE_PROGRAM_2 = """\
# Exemplo 2: pequeno laço com dependências e desvio
    ADDI R1, R0, 0     # i = 0
    ADDI R2, R0, 12    # limite = 12 (3 elementos: endereços 0,4,8)
LOOP:
    LD   R3, 0(R1)     # R3 = MEM[i]
    ADD  R4, R4, R3    # soma += R3
    ADDI R1, R1, 4     # i += 4
    BNE  R1, R2, LOOP  # enquanto i != limite
    NOP
"""


def create_simulator_from_assembly(asm: str) -> TomasuloSimulator:
    """Função de conveniência para a interface gráfica."""
    program = parse_assembly(asm)
    return TomasuloSimulator(program)



