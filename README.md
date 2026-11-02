# vault-watch
### Manual Download

1. Visit the [PaySim dataset on Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1).
2. Sign in to your Kaggle account (or create one if needed).
3. Click **Download** to download the dataset as a ZIP file.
4. Extract the ZIP file to obtain the CSV file:
   `PS_20174392719_1491204439457_log.csv`
5. In the root directory of the cloned VaultWatch repository, create the following folders if they do not already exist:

   ```bash
   mkdir -p data/raw
   ```

6. Move the extracted CSV file into `data/raw/`.

   Your project structure should look like:

   ```text
   vault-watch/
   ├── README.md
   └── data/
       └── raw/
           └── PS_20174392719_1491204439457_log.csv
   ```

7. Verify that the file is in the correct location:

   ```bash
   ls -lh data/raw/
   ```

**Important:** The full PaySim dataset must remain local and should never be committed to GitHub. Ensure `data/raw/` is included in `.gitignore`.