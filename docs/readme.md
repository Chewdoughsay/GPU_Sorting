# Investigație Sortare Paralelă — Task 2

## Detalii student

- **Nume:** Tudose Alexandru
- **Grupă:** TODO

---

## Configurație hardware

- **CPU:** TODO (ex: Apple M1 / Intel Core i7-...)
- **GPU (Colab):** NVIDIA T4 / A100 (verifică cu `nvidia-smi`)
- **RAM:** TODO
- **CUDA Version:** TODO (verifică cu `nvcc --version`)

---

## Task 2.1 — Timpi de execuție, comunicare și procesare

### Metodologie de măsurare

TODO — descrie cum măsori:
- Timp total (wall-clock)
- Timp kernel GPU (`cudaEvent_t`)
- Timp transfer H2D + D2H

### Tabele cu timpi

| Algoritm | N | CPU (s) | GPU Global (s) | GPU Shared (s) | Speedup (vs CPU) |
|---|---|---|---|---|---|
| Bitonic Sort | 1M | - | - | - | - |
| Shell Sort | 1M | - | - | - | - |
| Odd-Even Sort | 1M | - | - | - | - |
| Ranking Sort | 1M | - | - | - | - |
| Merge Sort | 1M | - | - | - | - |

### Analiză scalabilitate

TODO — repetă tabelul pentru N = 1K, 10K, 100K, 1M, 10M

---

## Task 2.2 — Discuție și analiză

### Comparație algoritmi

TODO

### Overhead comunicare

TODO

### Bottlenecks și optimizări

TODO

### Metrici — Speedup teoretic vs obținut

TODO — folosește formulele din lab 1 (Amdahl / Gustafson, efficiency, etc.)

---

*Document generat pentru tema de laborator GPU 2026.*
