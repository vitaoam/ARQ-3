import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from sim_core import (
    SAMPLE_PROGRAM_1,
    SAMPLE_PROGRAM_2,
    TomasuloSimulator,
    create_simulator_from_assembly,
)


class TomasuloApp:
    """Interface gráfica simples para o simulador de Tomasulo."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Simulador de Tomasulo - Trabalho 2")
        self.root.geometry("1200x700")

        # Estado do simulador
        self.sim: TomasuloSimulator = create_simulator_from_assembly(SAMPLE_PROGRAM_1)
        # Estado inicial de memória para os exemplos
        self._init_sample_memory()

        self.running: bool = False

        self._build_ui()
        self._refresh_all()

    # ------------------------------------------------------------------
    # Construção da interface
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        # Layout principal: esquerda (programa), direita (controle + tabelas)
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # ---------------------- Lado esquerdo: programa -----------------
        lbl_program = ttk.Label(left_frame, text="Programa MIPS simplificado")
        lbl_program.pack(anchor=tk.W)

        self.txt_program = tk.Text(left_frame, height=20, width=60, font=("Consolas", 10))
        self.txt_program.pack(fill=tk.BOTH, expand=True)
        self.txt_program.insert(tk.END, SAMPLE_PROGRAM_1)

        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, pady=5)

        btn_load = ttk.Button(btn_frame, text="Carregar / Reset a partir do texto", command=self.on_load_program)
        btn_load.pack(side=tk.LEFT, padx=2)

        btn_open = ttk.Button(btn_frame, text="Abrir arquivo .asm...", command=self.on_open_file)
        btn_open.pack(side=tk.LEFT, padx=2)

        # ---------------------- Lado direito: controles -----------------
        top_controls = ttk.Frame(right_frame)
        top_controls.pack(fill=tk.X)

        btn_step = ttk.Button(top_controls, text="Step (1 ciclo)", command=self.on_step)
        btn_step.pack(side=tk.LEFT, padx=2, pady=2)

        btn_run = ttk.Button(top_controls, text="Run", command=self.on_run)
        btn_run.pack(side=tk.LEFT, padx=2, pady=2)

        btn_pause = ttk.Button(top_controls, text="Pausar", command=self.on_pause)
        btn_pause.pack(side=tk.LEFT, padx=2, pady=2)

        btn_reset = ttk.Button(top_controls, text="Reset", command=self.on_reset)
        btn_reset.pack(side=tk.LEFT, padx=2, pady=2)

        btn_example2 = ttk.Button(
            top_controls,
            text="Carregar exemplo 2",
            command=self.on_load_example2,
        )
        btn_example2.pack(side=tk.LEFT, padx=2, pady=2)

        # Métricas
        metrics_frame = ttk.Frame(right_frame)
        metrics_frame.pack(fill=tk.X, pady=4)

        self.lbl_cycle = ttk.Label(metrics_frame, text="Ciclo: 0")
        self.lbl_cycle.pack(side=tk.LEFT, padx=5)

        self.lbl_committed = ttk.Label(metrics_frame, text="Commitadas: 0")
        self.lbl_committed.pack(side=tk.LEFT, padx=5)

        self.lbl_ipc = ttk.Label(metrics_frame, text="IPC: 0.00")
        self.lbl_ipc.pack(side=tk.LEFT, padx=5)

        self.lbl_bubbles = ttk.Label(metrics_frame, text="Ciclos de bolha: 0")
        self.lbl_bubbles.pack(side=tk.LEFT, padx=5)

        # Notebook com abas
        notebook = ttk.Notebook(right_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Aba: Instruções
        frame_instr = ttk.Frame(notebook)
        notebook.add(frame_instr, text="Instruções")
        self._build_instructions_tab(frame_instr)

        # Aba: RS / ROB
        frame_rs_rob = ttk.Frame(notebook)
        notebook.add(frame_rs_rob, text="RS / ROB")
        self._build_rs_rob_tab(frame_rs_rob)

        # Aba: Registradores / Memória
        frame_regs_mem = ttk.Frame(notebook)
        notebook.add(frame_regs_mem, text="Registradores / Memória")
        self._build_regs_mem_tab(frame_regs_mem)

    def _build_instructions_tab(self, parent: ttk.Frame) -> None:
        columns = ("id", "texto", "estagio", "issue", "exec", "write", "commit")
        tree = ttk.Treeview(parent, columns=columns, show="headings", height=15)

        tree.heading("id", text="ID")
        tree.heading("texto", text="Instrução")
        tree.heading("estagio", text="Estágio")
        tree.heading("issue", text="Issue")
        tree.heading("exec", text="Início Exec")
        tree.heading("write", text="Write")
        tree.heading("commit", text="Commit")

        tree.column("id", width=40, anchor=tk.CENTER)
        tree.column("texto", width=350)
        tree.column("estagio", width=100, anchor=tk.CENTER)
        tree.column("issue", width=60, anchor=tk.CENTER)
        tree.column("exec", width=80, anchor=tk.CENTER)
        tree.column("write", width=60, anchor=tk.CENTER)
        tree.column("commit", width=60, anchor=tk.CENTER)

        tree.pack(fill=tk.BOTH, expand=True)
        self.tree_instr = tree

    def _build_rs_rob_tab(self, parent: ttk.Frame) -> None:
        # RS na parte de cima
        frame_rs = ttk.LabelFrame(parent, text="Estações de Reserva")
        frame_rs.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        rs_columns = ("nome", "tipo", "busy", "op", "Vj", "Vk", "Qj", "Qk", "dest", "restante")
        self.tree_rs = ttk.Treeview(frame_rs, columns=rs_columns, show="headings", height=8)

        for col, text in zip(
            rs_columns,
            ["Nome", "Tipo", "Busy", "Op", "Vj", "Vk", "Qj", "Qk", "DestROB", "Rest. ciclos"],
        ):
            self.tree_rs.heading(col, text=text)

        self.tree_rs.column("nome", width=60, anchor=tk.CENTER)
        self.tree_rs.column("tipo", width=50, anchor=tk.CENTER)
        self.tree_rs.column("busy", width=50, anchor=tk.CENTER)
        self.tree_rs.column("op", width=60, anchor=tk.CENTER)
        self.tree_rs.column("Vj", width=80, anchor=tk.CENTER)
        self.tree_rs.column("Vk", width=80, anchor=tk.CENTER)
        self.tree_rs.column("Qj", width=60, anchor=tk.CENTER)
        self.tree_rs.column("Qk", width=60, anchor=tk.CENTER)
        self.tree_rs.column("dest", width=70, anchor=tk.CENTER)
        self.tree_rs.column("restante", width=80, anchor=tk.CENTER)

        self.tree_rs.pack(fill=tk.BOTH, expand=True)

        # ROB na parte de baixo
        frame_rob = ttk.LabelFrame(parent, text="Buffer de Reordenação (ROB)")
        frame_rob.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        rob_columns = (
            "idx",
            "seq",
            "busy",
            "ready",
            "state",
            "instr",
            "dest",
            "val",
            "branch",
            "addr",
        )
        self.tree_rob = ttk.Treeview(frame_rob, columns=rob_columns, show="headings", height=8)

        headers = [
            "Idx",
            "Seq",
            "Busy",
            "Ready",
            "Estado",
            "Instr",
            "DestReg",
            "Valor",
            "Branch",
            "Endereço",
        ]
        for col, text in zip(rob_columns, headers):
            self.tree_rob.heading(col, text=text)

        self.tree_rob.column("idx", width=40, anchor=tk.CENTER)
        self.tree_rob.column("seq", width=50, anchor=tk.CENTER)
        self.tree_rob.column("busy", width=50, anchor=tk.CENTER)
        self.tree_rob.column("ready", width=60, anchor=tk.CENTER)
        self.tree_rob.column("state", width=70, anchor=tk.CENTER)
        self.tree_rob.column("instr", width=200)
        self.tree_rob.column("dest", width=60, anchor=tk.CENTER)
        self.tree_rob.column("val", width=80, anchor=tk.CENTER)
        self.tree_rob.column("branch", width=80, anchor=tk.CENTER)
        self.tree_rob.column("addr", width=80, anchor=tk.CENTER)

        self.tree_rob.pack(fill=tk.BOTH, expand=True)

    def _build_regs_mem_tab(self, parent: ttk.Frame) -> None:
        # Registradores à esquerda
        frame_regs = ttk.LabelFrame(parent, text="Registradores")
        frame_regs.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.txt_regs = tk.Text(frame_regs, height=20, width=30, font=("Consolas", 10))
        self.txt_regs.pack(fill=tk.BOTH, expand=True)

        # Memória à direita
        frame_mem = ttk.LabelFrame(parent, text="Memória (endereços acessados)")
        frame_mem.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.txt_mem = tk.Text(frame_mem, height=20, width=40, font=("Consolas", 10))
        self.txt_mem.pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------------
    # Handlers de botões
    # ------------------------------------------------------------------
    def on_load_program(self) -> None:
        asm = self.txt_program.get("1.0", tk.END)
        try:
            self.sim = create_simulator_from_assembly(asm)
            self._init_sample_memory()
            self.running = False
            self._refresh_all()
        except Exception as e:
            messagebox.showerror("Erro ao carregar programa", str(e))

    def on_open_file(self) -> None:
        filename = filedialog.askopenfilename(
            title="Abrir arquivo de programa (.asm)",
            filetypes=[("Assembly", "*.asm *.txt"), ("Todos", "*.*")],
        )
        if not filename:
            return
        try:
            with open(filename, "r", encoding="utf-8") as f:
                text = f.read()
            self.txt_program.delete("1.0", tk.END)
            self.txt_program.insert(tk.END, text)
            self.on_load_program()
        except Exception as e:
            messagebox.showerror("Erro ao abrir arquivo", str(e))

    def on_step(self) -> None:
        self.sim.step_cycle()
        self._refresh_all()

    def on_run(self) -> None:
        if self.running:
            return
        self.running = True
        self._run_loop()

    def _run_loop(self) -> None:
        if not self.running:
            return
        if self.sim.finished:
            self.running = False
            self._refresh_all()
            return
        self.sim.step_cycle()
        self._refresh_all()
        # agenda próxima iteração
        self.root.after(200, self._run_loop)

    def on_pause(self) -> None:
        self.running = False

    def on_reset(self) -> None:
        self.sim.reset_state()
        self._init_sample_memory()
        self.running = False
        self._refresh_all()

    def on_load_example2(self) -> None:
        self.txt_program.delete("1.0", tk.END)
        self.txt_program.insert(tk.END, SAMPLE_PROGRAM_2)
        self.on_load_program()

    # ------------------------------------------------------------------
    # Atualização de UI
    # ------------------------------------------------------------------
    def _refresh_all(self) -> None:
        self._refresh_metrics()
        self._refresh_instructions()
        self._refresh_rs()
        self._refresh_rob()
        self._refresh_regs()
        self._refresh_mem()

    def _refresh_metrics(self) -> None:
        self.lbl_cycle.config(text=f"Ciclo: {self.sim.cycle}")
        self.lbl_committed.config(text=f"Commitadas: {self.sim.committed_instructions}")
        self.lbl_ipc.config(text=f"IPC: {self.sim.ipc:.2f}")
        self.lbl_bubbles.config(text=f"Ciclos de bolha: {self.sim.bubble_cycles}")

    def _refresh_instructions(self) -> None:
        self.tree_instr.delete(*self.tree_instr.get_children())
        for ins in self.sim.program:
            stage = self.sim.get_instruction_stage(ins)
            values = (
                ins.id,
                ins.text,
                stage,
                ins.issue_cycle if ins.issue_cycle is not None else "",
                ins.start_exec_cycle if ins.start_exec_cycle is not None else "",
                ins.write_result_cycle if ins.write_result_cycle is not None else "",
                ins.commit_cycle if ins.commit_cycle is not None else "",
            )
            self.tree_instr.insert("", tk.END, values=values)

    def _refresh_rs(self) -> None:
        self.tree_rs.delete(*self.tree_rs.get_children())
        for rs in self.sim.rs_add + self.sim.rs_mul + self.sim.rs_ls:
            values = (
                rs.name,
                rs.kind,
                "Sim" if rs.busy else "Não",
                rs.op or "",
                rs.Vj if rs.Vj is not None else "",
                rs.Vk if rs.Vk is not None else "",
                rs.Qj if rs.Qj is not None else "",
                rs.Qk if rs.Qk is not None else "",
                rs.dest_rob if rs.dest_rob is not None else "",
                rs.remaining_cycles,
            )
            self.tree_rs.insert("", tk.END, values=values)

    def _refresh_rob(self) -> None:
        self.tree_rob.delete(*self.tree_rob.get_children())
        for entry in self.sim.rob:
            instr_text = ""
            if entry.instr_id is not None and entry.instr_id in self.sim._instr_map:
                instr_text = self.sim._instr_map[entry.instr_id].text

            branch_info = ""
            if entry.is_branch:
                if entry.branch_actual_taken is None:
                    branch_info = "branch (?)"
                else:
                    branch_info = "taken" if entry.branch_actual_taken else "not-taken"

            addr_info = ""
            if entry.store_addr is not None:
                addr_info = str(entry.store_addr)

            values = (
                entry.index,
                entry.seq_num if entry.seq_num is not None else "",
                "Sim" if entry.busy else "Não",
                "Sim" if entry.ready else "Não",
                entry.state,
                instr_text,
                entry.dest_reg if entry.dest_reg is not None else "",
                entry.value if entry.value is not None else "",
                branch_info,
                addr_info,
            )
            self.tree_rob.insert("", tk.END, values=values)

    def _refresh_regs(self) -> None:
        self.txt_regs.config(state=tk.NORMAL)
        self.txt_regs.delete("1.0", tk.END)
        for i in range(0, 32, 4):
            parts = []
            for j in range(4):
                reg = i + j
                val = self.sim.registers[reg]
                parts.append(f"R{reg:02d}={val:6d}")
            self.txt_regs.insert(tk.END, "  ".join(parts) + "\n")
        self.txt_regs.config(state=tk.DISABLED)

    def _refresh_mem(self) -> None:
        self.txt_mem.config(state=tk.NORMAL)
        self.txt_mem.delete("1.0", tk.END)
        if not self.sim.memory:
            self.txt_mem.insert(tk.END, "(memória vazia)\n")
        else:
            for addr in sorted(self.sim.memory.keys()):
                val = self.sim.memory[addr]
                self.txt_mem.insert(tk.END, f"[{addr:4d}] = {val}\n")
        self.txt_mem.config(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # Estado inicial de exemplo
    # ------------------------------------------------------------------
    def _init_sample_memory(self) -> None:
        """
        Inicializa alguns valores em memória/registradores para facilitar a demonstração.
        Esses valores podem ser mudados à vontade pelo aluno.
        """
        # Zera tudo
        self.sim.memory.clear()
        # Pequeno vetor em memória: MEM[0] = 10, MEM[4] = 20, MEM[8] = 0
        self.sim.memory[0] = 10
        self.sim.memory[4] = 20
        self.sim.memory[8] = 0
        # Registradores base (por exemplo, R0 já é 0; usamos R1.. para laços)
        self.sim.registers[0] = 0
        self.sim.registers[1] = 0
        self.sim.registers[2] = 0
        self.sim.registers[3] = 0
        self.sim.registers[4] = 0

    # ------------------------------------------------------------------
    # Execução
    # ------------------------------------------------------------------
    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    app = TomasuloApp()
    app.run()


