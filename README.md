# Wood-cost-prediction
The README file has been prepared and includes details about data collection, the project approach, models used, and instructions for running the code.


## Overview
This project aims to predict the cost of wood based on various attributes such as wood type, dimensions, use case, and durability. The prediction leverages machine learning models to provide an accurate estimate of wood costs, which can be beneficial for suppliers, manufacturers, and customers.

---

## Data Collection
The dataset used for this project was synthetically generated to mimic real-world scenarios of wood cost variations. Here are the details of the data generation process:

1. **Attributes**:
   - **Wood Type**: 20 commonly used wood types (e.g., Teak, Oak, Pine).
   - **Length (cm)**: Random values between 80 and 250 cm.
   - **Width (cm)**: Random values between 10 and 60 cm.
   - **Thickness (cm)**: Random values between 1.0 and 4.5 cm.
   - **Use Case**: Examples include Furniture, Shelves, Tool Handles, Cutting Boards, etc.
   - **Durability**: Categorical values - Medium, High, Very High.
   - **Volume**: A derived feature calculated as `Length × Width × Thickness`.

2. **Target Variable**:
   - **Cost**: The cost of the wood is calculated using a formula that incorporates base cost, dimensions, and durability level.
   - Example Formula:
     ```
     cost = base_cost + (0.5 × Length) + (0.3 × Width) + (2 × Thickness)
     if durability == "High":
         cost *= 1.1
     elif durability == "Very High":
         cost *= 1.2
     ```

3. **Data Size**:
   - 200 rows of data were generated to ensure diversity and sufficient training data.


## Approach
The project followed these steps:

1. **Data Preprocessing**:
   - One-hot encoding for categorical features (e.g., Wood Type, Use Case, Durability).
   - Standardization of numerical features (Length, Width, Thickness, Volume).

2. **Feature Engineering**:
   - Created a new feature `Volume` to represent the product of length, width, and thickness.

3. **Model Selection**:
   - **Baseline Model**: Random Forest Regressor, chosen for its robustness and ability to handle non-linear relationships.
   - **Hyperparameter Tuning**: Used `GridSearchCV` to optimize hyperparameters such as the number of estimators, max depth, and minimum samples split.

4. **Evaluation Metrics**:
   - **Mean Squared Error (MSE)**: Measures average squared error.
   - **R-squared (R²)**: Evaluates how well the model explains the variance in the data.


## Models Used
### Random Forest Regressor
- A non-linear regression model that creates an ensemble of decision trees.
- Handles high-dimensional data and prevents overfitting using techniques like bootstrapping.

### Hyperparameter Tuning
- Conducted using `GridSearchCV` to find the optimal configuration:
  - Number of Trees (`n_estimators`): 100, 200, 300.
  - Maximum Depth (`max_depth`): None, 10, 20, 30.
  - Minimum Samples Split (`min_samples_split`): 2, 5, 10.
  - Minimum Samples Leaf (`min_samples_leaf`): 1, 2, 4.


## Results
- **Best Model**: Random Forest Regressor after hyperparameter tuning.
- **Performance Metrics**:
  - Mean Squared Error (MSE): ~[value]
  - R-squared (R²): ~[value]


## Running the Code
1. **Dependencies**:
   - Python 3.7+
   - Libraries: `pandas`, `numpy`, `scikit-learn`

2. **Steps**:
   - Clone the repository:
     ```bash
     git clone https://github.com/yourusername/wood-cost-prediction.git
     cd wood-cost-prediction
     ```
   - Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```
   - Run the script:
     ```bash
     python train_model.py
     ```

3. **Example Prediction**:
   Modify the input dictionary in the script:
   ```python
   example_input = {
       "Wood Type": "Teak",
       "Length (cm)": 150,
       "Width (cm)": 30,
       "Thickness (cm)": 2.5,
       "Use Case": "Furniture",
       "Durability": "High"
   }
   ```
   Run the script to see the predicted cost.


## Future Improvements
1. Collect real-world data to replace the synthetic dataset.
2. Explore advanced models like Gradient Boosting (e.g., XGBoost, LightGBM).
3. Add more features like region-specific pricing or seasonal variations.
4. Build a web interface for easier user interaction.



## Contributing
Feel free to submit pull requests or open issues for suggestions and improvements.


