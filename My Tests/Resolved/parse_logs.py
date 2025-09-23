import re
import csv
from pathlib import Path

root = Path(__file__).parent

def parse_cache_log(path: Path):
    text = path.read_text(encoding='utf-8', errors='ignore')
    # Statistics section
    m_hits = re.search(r'Total Hit: (\d+)', text)
    m_miss = re.search(r'Total Miss: (\d+)', text)
    m_total_access = re.search(r'Total Access: (\d+)', text)
    m_total_time = re.search(r'Total Time: (\d+)[\r\n]+\s*$', text)
    # Main Memory
    m_mm_reads = re.search(r'Main Memory[\s\S]*?Read Access: (\d+)', text)
    data = {
        'total_hit': int(m_hits.group(1)) if m_hits else None,
        'total_miss': int(m_miss.group(1)) if m_miss else None,
        'read_access': int(m_total_access.group(1)) if m_total_access else None,
        'total_time': int(m_total_time.group(1)) if m_total_time else None,
        'main_memory_read_access': int(m_mm_reads.group(1)) if m_mm_reads else None,
    }
    if data['read_access'] and data['total_time']:
        data['amat'] = data['total_time'] / data['read_access']
    return data

def write_csv(path: Path, header: list, rows: list):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)

def update_linesize_csv():
    csv_path = root / 'results_cache_linesize.csv'
    logs = [
        ('1', root / 'LS_1_TR_1_5.log'),
        ('2', root / 'LS_2_TR_1_5.log'),
        ('4', root / 'LS_4_TR_1_5.log'),
    ]
    header = ['line_size','read_access','total_hit','total_miss','hit_rate','main_memory_read_access','cache_read_time','main_memory_read_time','total_time','amat']
    rows = []
    for ls, p in logs:
        if not p.exists():
            continue
        d = parse_cache_log(p)
        hit_rate = d['total_hit']/d['read_access'] if d['read_access'] else ''
        rows.append([ls, d['read_access'], d['total_hit'], d['total_miss'], hit_rate, d['main_memory_read_access'], '', '', d['total_time'], d.get('amat')])
    write_csv(csv_path, header, rows)

def update_assoc_csv():
    csv_path = root / 'results_cache_associativity.csv'
    logs = [
        ('1', root / 'ASSOC_1_TR_CONFLIT.log'),
        ('2', root / 'ASSOC_2_TR_CONFLIT.log'),
        ('4', root / 'ASSOC_4_TR_CONFLIT.log'),
    ]
    header = ['associativity','read_access','total_hit','total_miss','hit_rate','main_memory_read_access','cache_read_time','main_memory_read_time','total_time','amat']
    rows = []
    for assoc, p in logs:
        if not p.exists():
            continue
        d = parse_cache_log(p)
        hit_rate = d['total_hit']/d['read_access'] if d['read_access'] else ''
        rows.append([assoc, d['read_access'], d['total_hit'], d['total_miss'], hit_rate, d['main_memory_read_access'], '', '', d['total_time'], d.get('amat')])
    write_csv(csv_path, header, rows)

if __name__ == '__main__':
    update_linesize_csv()
    update_assoc_csv()
    print('CSV atualizados em:', root)

