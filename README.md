# Automated Oil Spill Detection using AIS and Satellite Data

An AI-powered system that integrates AIS (Automatic Identification System) data with satellite imagery to automatically detect oil spills and identify vessels in distress in real-time.

## 🚀 Overview

This project aims to develop an end-to-end solution that:
- Detects anomalies in AIS data (like erratic speed, abrupt stops, abnormal course changes).
- Correlates suspicious behavior with satellite data to confirm oil spills.
- Provides early alerts for potential environmental hazards and maritime accidents.

## 📌 Key Features

- ✅ Real-time anomaly detection using vessel AIS data.
- 🛰️ Satellite image analysis using SAR data from the Copernicus Data Space Ecosystem.
- ⚠️ Identification of distress zones and spill-prone areas.
- 📈 Visualization and report generation.
- 🔐 Secure and scalable architecture for deployment.

## 🗃️ Dataset

- **AIS Data**: Includes MMSI, SOG (Speed Over Ground), COG (Course Over Ground), Draught, Latitude, Longitude, and Timestamp.
- **Satellite Imagery**: JPEG SAR images from [Copernicus Open Access Hub](https://dataspace.copernicus.eu/).
- **Labels**: Oil spill / No oil spill annotations for supervised training.

> _Note: For privacy and size reasons, raw datasets are not included in the repo. Please refer to the `data/README.md` for sample links and preprocessing scripts._

## 🛠️ Tech Stack

| Category            | Tools & Frameworks                        |
|---------------------|-------------------------------------------|
| Programming Language| Python                                    |
| Data Processing     | Pandas, NumPy                             |
| Machine Learning    | Scikit-learn, XGBoost                     |
| Deep Learning       | TensorFlow, Keras, CNN for image analysis|
| Visualization       | Matplotlib, Seaborn                       |
| GIS Tools           | Rasterio, OpenCV                          |
| Deployment          | Streamlit, Flask (for demo interfaces)    |


## 📊 Model Performance

| Model        | AIS Accuracy | Image Accuracy | Combined Model F1 |
| ------------ | ------------ | -------------- | ----------------- |
| RandomForest | 87%          | N/A            | 82%               |
| CNN (SAR)    | N/A          | 90%            | 85%               |
| Ensemble     | 89%          | 91%            | **91.4%**         |

## 💡 Future Work

* Add integration with vessel registry APIs to track responsible parties.
* Improve temporal correlation between AIS data and satellite imagery.
* Deploy on cloud with real-time streaming input (Kafka + FastAPI).


## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🙋‍♀️ Acknowledgments

* European Space Agency for the Copernicus Sentinel data.
* [MarineCadastre.gov](https://marinecadastre.gov/) for AIS open datasets.
* Researchers and open-source contributors in marine environmental AI.

