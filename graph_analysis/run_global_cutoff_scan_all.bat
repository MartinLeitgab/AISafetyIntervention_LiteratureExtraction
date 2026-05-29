@echo off
cd /d C:\Users\malei\0_project_work\eleutherAI_SOAR_step1knowledgegraphcreation\AISafetyIntervention_LiteratureExtraction\graph_analysis
set PYTHONIOENCODING=utf-8

echo === RUN cutoff=0.80 (clustering) ===
python phase2_step4_global_cutoff_scan.py --cutoff 0.80 1>logfiles/global_cutoff_0.80_clust.log 2>&1
echo === RUN cutoff=0.80 (F1+Hamming) ===
python phase2_step4_F1_global_threelevel.py --cutoff 0.80 1>logfiles/global_cutoff_0.80_f1.log 2>&1

echo === RUN cutoff=0.90 (clustering) ===
python phase2_step4_global_cutoff_scan.py --cutoff 0.90 1>logfiles/global_cutoff_0.90_clust.log 2>&1
echo === RUN cutoff=0.90 (F1+Hamming) ===
python phase2_step4_F1_global_threelevel.py --cutoff 0.90 1>logfiles/global_cutoff_0.90_f1.log 2>&1

echo === RUN cutoff=0.95 (clustering) ===
python phase2_step4_global_cutoff_scan.py --cutoff 0.95 1>logfiles/global_cutoff_0.95_clust.log 2>&1
echo === RUN cutoff=0.95 (F1+Hamming) ===
python phase2_step4_F1_global_threelevel.py --cutoff 0.95 1>logfiles/global_cutoff_0.95_f1.log 2>&1

echo === ALL DONE ===
