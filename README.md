# Aplikasi Screening Respons Acidizing

Folder ini adalah paket Streamlit inference yang bisa dicopy dan dijalankan
sendiri. Kode aplikasi, model `.joblib`, metadata model, reference frame untuk
flag review, dependency list, dan contoh CSV semuanya ada di folder ini.

## Isi Paket

```text
streamlit-inference/
  streamlit_app.py
  model_service.py
  requirements.txt
  README.md
  models/
    rf_50q/
      random_forest.joblib
      reference_frame.csv
      model_metadata.json
  samples/
    batch_template.csv
```

## Model

Model aktif:

- `RF 50% median`: `models/rf_50q/random_forest.joblib`

Batas kelas:

- `low_gain`: gain <= `0.768627451`
- `high_gain`: gain > `0.768627451`

## Menjalankan Dengan uv

```bash
uv venv .venv
.venv\Scripts\activate
uv pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

## Output

Aplikasi menerima input mentah bergaya data training, menghitung fitur dasar
dan domain-engineered yang sama untuk inference, lalu mengembalikan:

- prediksi `low_gain` / `high_gain`
- label prediksi Bahasa Indonesia
- confidence
- flag perlu review
- alasan
- probabilitas gain rendah dan gain tinggi

## Kolom Input

CSV harus memuat:

- `Well`
- `Tipe Acid`
- `Sandstone`
- `Siltstone`
- `Claystone`
- `Shale`
- `Coal`
- `Limestone`
- `IPR Vogel sebelum pengasaman (BFPD)`
- `Batas Reservoar` dalam ft

Interval yang digunakan model diparsing dari `Batas Reservoar` dalam ft.

`Tipe Acid` terbatas pada training dan mendukung:

- `Mud Acidizing 12% HCl : 3% HF`
- `Matrix Acidizing 15% HCl`

Contoh CSV tersedia di `samples/batch_template.csv`.

## Interpretasi

Aplikasi ini adalah alat screening dan pendukung keputusan untuk engineer.
Model menggunakan batas median 50% untuk membedakan gain rendah dan tinggi
pada populasi pemodelan setelah eksklusi IQR. Prediksi tetap perlu ditinjau
bersama konteks reservoir, produksi, petrofisika, dan operasi.
