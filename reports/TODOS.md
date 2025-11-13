### TODOS
| # | Task                                                                                                                    | Done?                                       |
| - | ----------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| 1 | Build Mel-spectrogram pipeline (librosa.melspectrogram → power_to_db; normalize per-sample)                             | ✅                                           |
| 2 | 2D CNN baseline (train 3–4 conv blocks, report val/test + confusion matrix)                                             | ⚠️ - Need to Review                          |
| 3 | **MFCC vs Mel comparison** (side-by-side metrics & confusion matrices in one notebook or table)                         | ❌                                           |
| 4 | **CNN–LSTM hybrid** (CNN feature extractor → reshape to sequences → LSTM)                                               | ❌                                           |
| 5 | **Noise-robustness experiment** (add 0/10/30/50% Gaussian noise; test frozen models)                                    | ❌                                           |
| 6 | **Interpretability** (t-SNE embeddings + Grad-CAM heatmaps)                                                             | ❌                                           |
| 7 | **Live demo** (Gradio or Streamlit app for real-time prediction from uploaded audio)                                    | ❌                                           |
| 8 | **Final write-up housekeeping** (document splits, early stopping, tuned hyperparams, validation F1-driven model choice) | ⚠️ Partial — needs formal summary in report |

---

