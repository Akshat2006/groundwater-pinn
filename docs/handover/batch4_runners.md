# Batch 4: how to run your notebook (one page)

You are helping train the project's final models on Google Colab. You only press buttons; nothing needs to be understood or changed.

## What you get from the project lead
1. `pinn_code.zip` (about 9 MB)
2. **One** notebook: `colab_real_batch4_L0.ipynb`, `_L1` or `_L2`, with each person getting a different one

| Notebook | Runs | Time |
|---|---|---|
| L0 | 8 | about 3.5 hours |
| L1 | 10 | about 4.5 hours |
| L2 | 10 | about 4.5 hours |

## Steps
1. **Google Drive:** make a folder called `pinn` in *My Drive* and upload `pinn_code.zip` into it, so you have `My Drive/pinn/pinn_code.zip`.
2. **Open the notebook:** go to colab.research.google.com → *File → Upload notebook* → choose your notebook.
3. **GPU:** *Runtime → Change runtime type → T4 GPU → Save*.
4. **Start:** *Runtime → Run all*. When asked, allow access to Google Drive (pick your own account).
5. **Wait.** Each run prints training progress and takes 20–35 minutes. Lines like `already done: ...` are normal after a restart.
6. **Finish:** when the runs are done (or if you have to stop), run the **last cell**. It writes `My Drive/pinn/real4_LN_results.zip`.
7. **Send that zip** to the project lead (Drive share link or download and send).

## Keep it running
- Laptop plugged in, sleep turned off (*Windows Settings → System → Power → Screen and sleep → "Never"* when plugged in), and the Colab tab left open. The screen may turn off; the laptop must not sleep.
- **If Colab disconnects or says you've hit a usage limit:** wait (or come back the next day), open the same notebook and *Run all* again. Finished runs are kept on your Drive and skipped.

## Rules
- Don't edit any cell, file or setting. Don't run other notebooks or scripts.
- If something prints a red error and stops, take a screenshot of the last 20 lines and send it to the project lead.
