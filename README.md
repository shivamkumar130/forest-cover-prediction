# Forest Cover Type Prediction

This project aims to build a machine learning system to predict forest cover types based on cartographic variables from the Roosevelt National Forest of northern Colorado.

## Project Structure

## Dataset

The dataset contains cartographic information about 30x30 meter cells in the Roosevelt National Forest. The features include:

- Elevation: Elevation in meters
- Aspect: Aspect in degrees azimuth
- Slope: Slope in degrees
- Horizontal_Distance_To_Hydrology: Horz Dist to nearest surface water features
- Vertical_Distance_To_Hydrology: Vert Dist to nearest surface water features
- Horizontal_Distance_To_Roadways: Horz Dist to nearest roadway
- Hillshade_9am: Hillshade index at 9am, summer solstice (0 to 255 index)
- Hillshade_Noon: Hillshade index at noon, summer solstice (0 to 255 index)
- Hillshade_3pm: Hillshade index at 3pm, summer solstice (0 to 255 index)
- Horizontal_Distance_To_Fire_Points: Horz Dist to nearest wildfire ignition points
- Wilderness_Area: 4 binary columns (0 = absence or 1 = presence)
- Soil_Type: 40 binary columns (0 = absence or 1 = presence)
- Cover_Type: Forest Cover Type designation (1-7)

## Forest Cover Types

1. Spruce/Fir
2. Lodgepole Pine
3. Ponderosa Pine
4. Cottonwood/Willow
5. Aspen
6. Douglas-fir
7. Krummholz

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Place your Excel data file in `data/raw/forest_cover_data.xlsx`

## Usage

1. Preprocess the data: `python src/data_processing.py`
2. Train the models: `python src/modeling.py`
3. Generate visualizations: `python src/visualization.py`
4. Explore the analysis in the Jupyter notebook: `notebooks/forest_cover_analysis.ipynb`

## Results

The project achieves an accuracy of X.X% in predicting forest cover types using a Random Forest classifier. The most important features for prediction are elevation, horizontal distance to roadways, and horizontal distance to fire points.

## License

This project is licensed under the MIT License.
