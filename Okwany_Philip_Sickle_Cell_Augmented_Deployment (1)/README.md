# Sickle cell image analysis

**Okwany Philip · Augmented model deployment**

The app accepts a microscope image and displays the field prediction, detector score, candidate boxes, and segmentation overlay. It also exports the binary mask as a PNG.

## Deploy with a shareable link

1. Extract the download on your computer. Open the extracted folder so you can see `app.py`, `inference.py`, `requirements.txt`, `model_config.json`, the two `.pt` files, and two example JPEGs.
2. Create a repository named `sickle-cell-augmented` on GitHub. Choose **Public** if viewers should be able to open your demo without requesting access. Initialize it with a README. Open **Add file → Upload files** and upload the contents of the extracted folder to the repository root. Commit the upload.
3. Open Streamlit Community Cloud and connect your GitHub account. Click **Create app**, then **Yup, I have an app**. Enter your repository, its branch (usually `main`), and `app.py` as the main file. Use Python **3.12** in **Advanced settings**.
4. Click **Deploy**. Wait for the app page to open, then copy its `streamlit.app` URL. Open the link and run both held-out samples before sharing it.

GitHub repository creation and uploads are described in the [GitHub guide](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository). The deployment screen is described in the [Streamlit guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

## Try a prediction

Choose **Upload an image** and select a JPEG, PNG or TIFF from your computer. Click **Analyze image**. Use **Try a held-out sample** if you want an example with a known Positive or Negative reference label. The gold boxes are cell candidates and the green areas are predicted masks. The detector score comes from the strongest candidate in the field.

The first analysis loads the saved checkpoints and downloads the project's public COCO backbone from PyTorch. Later analyses reuse the loaded model. If the app has hibernated, open its link again and allow it to start.

## Model and evaluation

The repository contains the trained Augmented detector and U-Net checkpoints from the report. The frozen backbone is identical to the one in that model export. The saved detector cutoff is 0.414882 and the mask cutoff is 0.65. The full 82-image benchmark gave 87.80% image accuracy and 87.12% balanced accuracy. The model missed 7 of the 61 positive fields. Use this app as a microscopy research demonstration.

To rerun the full labeled evaluation, use `Sickle_Cell_Colab_Test_Guide.ipynb` with the matching model and Test ZIPs. The two app examples are held-out images from the [Tushabe et al. dataset](https://www.kaggle.com/datasets/florencetushabe/sickle-cell-disease-dataset).

## Run locally

With Python 3.12 installed, open a terminal in the extracted folder and run these commands. The app opens at `http://localhost:8501`.

```text
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
