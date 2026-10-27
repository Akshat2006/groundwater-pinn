# Batch 5: how to run the M8 notebook (one page)

The steps are the same as batch 4. There is one notebook and it needs the **new** code zip.

## What you get from the project lead
1. `pinn_code.zip` (new, about 9 MB). **Replace** any older copy in `My Drive/pinn/`.
2. `colab_real_batch5_m8.ipynb`: 12 runs, about 4 hours.

## Steps
1. Put `pinn_code.zip` in `My Drive/pinn/`, replacing the old one.
2. Go to colab.research.google.com → *File → Upload notebook* → choose `colab_real_batch5_m8.ipynb`.
3. *Runtime → Change runtime type → T4 GPU → Save*.
4. *Runtime → Run all*, and allow Google Drive access.
5. Wait.
   - The first 3 runs ("nophys") take about 5-10 minutes each. The others take 20-35 minutes each.
   - Lines like `already done: ...` are normal after a restart.
6. Run the **last cell**. It writes `My Drive/pinn/real5_m8_results.zip`. Send that zip to the project lead.

## Keep it running
- Keep the laptop plugged in with sleep turned off, and leave the Colab tab open.
- If Colab disconnects, open the same notebook and *Run all* again. Finished runs are skipped.

## Rules
- Don't edit anything.
- If you see an error saying `old pinn_code.zip`, the Drive still has the old zip. Replace it and *Run all* again.
- For any other red error that stops the notebook, send a screenshot of the last 20 lines.
