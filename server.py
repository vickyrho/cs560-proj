import pandas as pd
from IPython.display import display
from datetime import datetime
from z3 import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.decomposition import PCA
import hdbscan
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

def preprocess(cars_df):
    current_year = datetime.now().year
    cars_df['Age_of_Car'] = current_year - cars_df['Year']
    cars_df.drop(columns=['Year'], inplace=True)
    cars_df['Mileage'] = cars_df['Mileage'].str.replace(r'[^\d.]+', '', regex=True).astype(float)
    cars_df['Engine'] = cars_df['Engine'].str.replace(r'[^\d.]+', '', regex=True).astype(float)
    cars_df['Power'] = cars_df['Power'].str.replace(r'[^\d.]+', '', regex=True).astype(float)
    cars_df['Transmission'] = cars_df['Transmission'].map({'Manual': 0, 'Automatic': 1})
    cars_df['Fuel_Type'] = cars_df['Fuel_Type'].map({'Petrol': 0, 'Diesel': 1})
    cars_df['Owner_Type'] = cars_df['Owner_Type'].map({'First': 1, 'Second': 2, 'Third': 3, 'Fourth': 4})
    
    # cars_df['Price_per_km'] = cars_df['Kilometers_Driven'] / cars_df['Engine']
    # cars_df['Age_Mileage'] = cars_df['Age_of_Car'] * cars_df['Kilometers_Driven']
    

    cars_df.dropna(inplace=True)

    return cars_df
    # cars_df['Mileage'].fillna(cars_df['Mileage'].median(), inplace=True)  # Replace NaN in 'Year' with median
    # cars_df['Owner_Type'].fillna(cars_df['Owner_Type'].mode()[0], inplace=True)  # Replace NaN in 'Owner_Type' with mode
    # cars_df['Engine'].fillna(cars_df['Engine'].mean(), inplace=True)

    # cars_df = pd.get_dummies(cars_df, columns=['Location'], drop_first=True)

# Load the datasets
cars_df = pd.read_csv('used_cars_india.csv')
pd.set_option('display.max_columns', None)
# Pretty print the head of the dataframe
cars_df = cars_df.drop(cars_df.columns[0], axis=1)
columns_to_drop = ['New_Price', 'Location']
cars_df = cars_df.drop(columns=columns_to_drop)


cars_df = preprocess(cars_df)
display(cars_df.head())
print(f"DataFrame length: {len(cars_df)}")


# In[231]:


max_price = cars_df['Price'].max()
min_price = cars_df['Price'].min()

print(f"Maximum Price: {max_price}")
print(f"Minimum Price: {min_price}")


# In[232]:


lowest_priced_cars = cars_df.nsmallest(5, 'Price')
display(lowest_priced_cars)


# In[233]:



Q1 = cars_df['Price'].quantile(0.25)
Q3 = cars_df['Price'].quantile(0.75)
IQR = Q3 - Q1

# Define upper bound
upper_bound = Q3 + 1.5 * IQR
# print(f"Upper Bound for Price: {upper_bound}")

min_reasonable_price = cars_df['Price'].quantile(0.5) * 0.85
print(min_reasonable_price)
# Filter out expensive outliers
cars_df = cars_df[cars_df['Price'] <= upper_bound]

# Display the filtered DataFrame
# print("Filtered DataFrame (After Removing Expensive Outliers):")
# print(filtered_cars_df)

# Plot a boxplot for the filtered dataset
plt.boxplot(cars_df['Price'])
plt.title('Boxplot of Car Prices After Removing Expensive Outliers')
plt.ylabel('Price')
plt.show()


# In[234]:



# Plot the price distribution
plt.figure(figsize=(10, 6))
plt.hist(cars_df['Price'], bins=20, edgecolor='black', alpha=0.7, density=True)

# Convert density to percentage
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y * 100)}%'))

# Set x-axis ticks in intervals of 10
plt.xticks(range(0, int(cars_df['Price'].max()) + 10, 5))

# Add titles and labels
plt.xlim(right=20)
plt.title('Price Distribution (Percentage)', fontsize=16)
plt.xlabel('Price (in lakhs)', fontsize=12)
plt.ylabel('Percentage', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# In[235]:


display(cars_df.head())


# In[236]:



def custom_transform(cars_df, fit = False):
    # Column to drop
    columndrop = 'Name'

    # Drop the column only if it exists
    if columndrop in cars_df.columns:
        cars_df = cars_df.drop(columns=[columndrop])
    columns_to_scale = ['Age_of_Car', 'Kilometers_Driven', 'Mileage', 'Engine', 'Power', 'Seats', 'Price']

    if fit:
        cars_df[columns_to_scale] = scaler.fit_transform(cars_df[columns_to_scale])
    else:
        cars_df[columns_to_scale] = scaler.transform(cars_df[columns_to_scale])        

    return cars_df

scaler = StandardScaler()
cols = [col for col in cars_df.columns if col != 'Price'] + ['Price']
cars_scaled_df = custom_transform(cars_df, fit = True)

print("DataFrame after Standard Scaling:")
display(cars_scaled_df.head())


# In[237]:


def reverse_transform(cars_df):
    # Define the columns to reverse scale
    columns_to_scale = ['Age_of_Car', 'Kilometers_Driven', 'Mileage', 'Engine', 'Power', 'Seats', 'Price']
    # Reverse the scaling
    cars_df[columns_to_scale] = scaler.inverse_transform(cars_df[columns_to_scale])
    # display(cars_df)
    return cars_df


# In[238]:


print(cars_scaled_df.shape)


# In[239]:



# Split the data into features and target
X_cars = cars_scaled_df.drop('Price', axis=1)  # Features
y_cars = cars_scaled_df['Price']  # Target variable (Price)

# Split the dataset into training and testing sets
X_train_cars, X_test_cars, y_train_cars, y_test_cars = train_test_split(X_cars, y_cars, test_size=0.3, random_state=0)


# In[240]:


# Initialize the model
linear_reg_model = LinearRegression()

# Train the model
linear_reg_model.fit(X_train_cars, y_train_cars)

# Make predictions on the test set
y_pred_cars = linear_reg_model.predict(X_test_cars)

# Calculate performance metrics
rmse = mean_squared_error(y_test_cars, y_pred_cars, squared=False)
r2 = r2_score(y_test_cars, y_pred_cars)

# Print out the RMSE and R-squared values
print(f"Model - RMSE: {rmse}")
print(f"Model - R² Score: {r2}")


# In[241]:


# # Initialize and train the Random Forest Regressor
# rf = RandomForestRegressor(random_state=0, n_estimators=100)
# rf.fit(X_train_cars, y_train_cars)

# # Make predictions
# y_pred_cars = rf.predict(X_test_cars)

# # Evaluate the model
# rmse = mean_squared_error(y_test_cars, y_pred_cars, squared=False)
# r2 = r2_score(y_test_cars, y_pred_cars)

# print(f"Model - RMSE: {rmse}")
# print(f"Model - R² Score: {r2}")


# In[242]:


# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test_cars, y_pred_cars, alpha=0.5)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual Prices vs Predicted Prices')
plt.plot([y_test_cars.min(), y_test_cars.max()], [y_test_cars.min(), y_test_cars.max()], 'k--', lw=2)
plt.show()


# In[243]:


# data = {
#     'Name': [
#         'Hyundai Creta 1.6 CRDi SX', 'Honda City i-DTEC VX', 
#         'Maruti Suzuki Swift VDI', 'Toyota Innova Crysta 2.4 GX',
#         'Ford EcoSport Titanium 1.5'
#     ],
#     'Kilometers_Driven': [41000, 60000, 35000, 72000, 45000],
#     'Fuel_Type': ['Diesel', 'Diesel', 'Diesel', 'Diesel', 'Petrol'],
#     'Transmission': ['Manual', 'Manual', 'Manual', 'Manual', 'Automatic'],
#     'Owner_Type': ['First', 'First', 'Second', 'First', 'First'],
#     'Mileage': ['19.67 mpg', '23.67 mpg', '22.7 mpg', '15.1 mpg', '17.0 mpg'],
#     'Engine': ['1582 hp', '1498 cc', '1248 cc', '2393 cc', '1496 cc'],
#     'Power': ['126.2 hp', '100.0 hp', '74.0 hp', '148.0 hp', '123.0 hp'],
#     'Seats': [5, 5, 5, 7, 5],
#     'Year': [2015, 2012, 2017, 2015, 2020],
#     'Price': 0
# }

# # Convert to DataFrame
# sample_df = pd.DataFrame(data)

# # Display the sample DataFrame
# df = preprocess(sample_df)
# print("before")
# display(df.head())
# df = custom_transform(df)
# print("after")
# df = df.drop(columns=['Price'])


# In[244]:


# sample_y_pred_cars = linear_reg_model.predict(df)
# df['Price'] = sample_y_pred_cars
# display(df)


# In[245]:


# cars_df_untransform = reverse_transform(df)
# display(cars_df_untransform)


# In[246]:


# base_point = {
#   "Name": 'Honda City 1.5 V AT Sunroo',
#   "Kilometers_Driven": 60000,
#   "Fuel_Type": "Petrol",
#   "Transmission": "Automatic",
#   "Owner_Type": "First",
#   "Mileage": "16.8 kmpl",
#   "Engine": "1497 CC",
#   "Power": "116.3 bhp",
#   "Seats": 5,  
#   "Price": 4.49,
#   "Year": 2012
# }
# tsample_df = pd.DataFrame([base_point])
# df = preprocess(tsample_df.copy())
# df = custom_transform(df)
# display(df)


# In[247]:


# max_price_row_df = cars_scaled_df.loc[[cars_df['Price'].idxmax()]]
# min_price_row_df = cars_scaled_df.loc[[cars_df['Price'].idxmin()]]

# display(max_price_row_df)
# display(min_price_row_df)

# min_price = reverse_transform(min_price_row_df)
# max_price = reverse_transform(max_price_row_df)

# display(max_price_row_df)
# display(min_price_row_df)


# In[248]:


# Define the features and coefficients
features = [
    "Kilometers_Driven", "Fuel_Type", "Transmission", "Owner_Type",
    "Mileage", "Engine", "Power", "Seats", "Age_of_Car"
]

print(X_train_cars.columns)

coefficients = linear_reg_model.coef_
intercept = linear_reg_model.intercept_
# importances = rf.feature_importances_
# feature_imp_df = pd.DataFrame({'Feature': features, 'Gini Importance': importances}).sort_values('Gini Importance', ascending=True) 
# print(feature_imp_df)

# Create a bar plot for feature importance
# plt.figure(figsize=(8, 4))
# plt.barh(feature_names, importances, color='skyblue')
# plt.xlabel('Gini Importance')
# plt.title('Feature Importance - Gini Importance')
# plt.gca().invert_yaxis()  # Invert y-axis for better visualization
# plt.show()

# Combine features and coefficients into a dictionary
feature_coeff_dict = dict(zip(features, coefficients))

# Sort the features by the absolute value of their coefficients
sorted_features = sorted(feature_coeff_dict.items(), key=lambda x: abs(x[1]), reverse=True)

# Print the sorted feature-importance pairs
sum = 0
print("Feature Coefficients:")
for feature, coef in sorted_features:
    print(f"{feature}: {coef:.4f}")
    sum = sum + coef
    
print(sum)


# In[249]:


# Define the absolute value function
def z3_abs(x):
    return If(x >= 0, x, -x)


# In[250]:


existing_solutions = pd.DataFrame(columns=cols)
existing_solutions = existing_solutions.drop(columns=['Name'])
iterations = 1
max_iterations = 100
# max_iterations = 20


# In[251]:


def z3_to_solution(km_driven, fuel_type, transmission, owner_type, mileage, engine, power, seats, age):
    # Assuming coefficients and intercept for prediction are defined
    pr = (
        coefficients[0] * km_driven +
        coefficients[1] * fuel_type +
        coefficients[2] * transmission +
        coefficients[3] * owner_type +
        coefficients[4] * mileage +
        coefficients[5] * engine +
        coefficients[6] * power +
        coefficients[7] * seats +
        coefficients[8] * age +
        intercept
    )

    # Create a DataFrame with the input and the predicted price
    solution_df = pd.DataFrame([{
        'Kilometers_Driven': km_driven,
        'Fuel_Type': fuel_type,
        'Transmission': transmission,
        'Owner_Type': owner_type,
        'Mileage': mileage,
        'Engine': engine,
        'Power': power,
        'Seats': seats,
        'Age_of_Car': age,
        'Price': pr
    }])
    # Apply any custom untransform logic if necessary
    untransformed_solution = reverse_transform(solution_df)
    untransformed_solution_df = pd.DataFrame(untransformed_solution)

    return pr, untransformed_solution_df


# In[252]:



# Initialize the Z3 solver
solver = Solver()
solver.set("timeout", 1000)

# Define Z3 variables
km_driven_z3 = Real('Kilometers_Driven')
fuel_type_z3 = Real('Fuel_Type')  # 0 for Diesel, 1 for Petrol
transmission_z3 = Real('Transmission')  # 0 for Manual, 1 for Automatic
owner_type_z3 = Real('Owner_Type')  # 1 for First Owner, 2 for Second Owner
mileage_z3 = Real('Mileage')
engine_z3 = Real('Engine')
power_z3 = Real('Power')
seats_z3 = Real('Seats')
age_of_car_z3 = Real('Age_of_Car')  # Instead of Year, we use Age_of_Car

# Model coefficients and intercept
coefficients = linear_reg_model.coef_

intercept = linear_reg_model.intercept_

print(existing_solutions)

# Prediction equation based on the linear regression model
prediction = (
    coefficients[0] * km_driven_z3 +
    coefficients[1] * fuel_type_z3 +
    coefficients[2] * transmission_z3 +
    coefficients[3] * owner_type_z3 +
    coefficients[4] * mileage_z3 +
    coefficients[5] * engine_z3 +
    coefficients[6] * power_z3 +
    coefficients[7] * seats_z3 +
    coefficients[8] * age_of_car_z3 +  # Using Age_of_Car as Year proxy
    intercept
)

# print(prediction)


# In[253]:


def append_as_unique_ints(df, tdf):
    tdf = tdf.astype(int)
    row = tdf.iloc[0]
    
    # exclude the price column from the comparison
    columns_to_compare = df.columns[:-1]
    
    # if the row (excluding the price column) is unique in the DataFrame
    if not ((df[columns_to_compare] == row[columns_to_compare]).all(axis=1)).any():
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        # df.to_csv('solutions.csv', index=False)
    
    return df


# In[254]:


# cars_df['Price_per_km'] = cars_df['Kilometers_Driven'] / cars_df['Engine']
# cars_df['Age_Mileage'] = cars_df['Age_of_Car'] * cars_df['Mileage']

# delta = {
#     'Kilometers_Driven': 1000,       # Significant difference in kilometers driven
#     'Fuel_Type': 1,               # Small difference, or use strict equality
#     'Transmission': 1,              # Strict equality for binary variables
#     'Owner_Type': 1,                # At least one ownership difference
#     'Mileage': 2.0,                 # Noticeable difference in mileage (kmpl)
#     'Engine': 100,                  # Significant difference in engine size (cc)
#     'Power': 10,                    # Difference in power (bhp)
#     'Seats': 0,                     # Strict equality for seat count
#     'Age_of_Car': 1,                # Difference in years of age
#     'Age_Mileage': 10000,             # Difference in derived feature
#     'Price': 0.5          # Difference in predicted price (lakhs)
# }

delta = {
    'Kilometers_Driven': 0.1,       # Significant difference in kilometers driven
    'Fuel_Type': 0.1,               # Small difference, or use strict equality
    'Transmission': 0.1,              # Strict equality for binary variables
    'Owner_Type': 0.1,                # At least one ownership difference
    'Mileage': 0.1,                 # Noticeable difference in mileage (kmpl)
    'Engine': 0.1,                  # Significant difference in engine size (cc)
    'Power': 0.1,                    # Difference in power (bhp)
    'Seats': 0.1,                     # Strict equality for seat count
    'Age_of_Car': 0.1,                # Difference in years of age
    'Price': 0.1         # Difference in predicted price (lakhs)
}


lower_bound = {
    'Kilometers_Driven': 10000,       # Significant difference in kilometers driven
    'Fuel_Type': 0,               # Small difference, or use strict equality
    'Transmission': 0,              # Strict equality for binary variables
    'Owner_Type': 1,                # At least one ownership difference
    'Mileage': 10.0,                 # Noticeable difference in mileage (kmpl)
    'Engine': 700,                  # Significant difference in engine size (cc)
    'Power': 50,                    # Difference in power (bhp)
    'Seats': 4,                     # Strict equality for seat count
    'Age_of_Car': 3,                # Difference in years of age
    'Price': 1.0,          # Difference in predicted price (lakhs)
}

upper_bound = {
    'Kilometers_Driven': 150000,       # Significant difference in kilometers driven
    'Fuel_Type': 1,               # Small difference, or use strict equality
    'Transmission': 1,              # Strict equality for binary variables
    'Owner_Type': 4,                # At least one ownership difference
    'Mileage': 40.0,                 # Noticeable difference in mileage (kmpl)
    'Engine': 3000,                  # Significant difference in engine size (cc)
    'Power': 300,                    # Difference in power (bhp)
    'Seats': 7,                     # Strict equality for seat count
    'Age_of_Car': 10,                # Difference in years of age
    'Price': 15.0,          # Difference in predicted price (lakhs)
}

upper_bound = custom_transform(pd.DataFrame([upper_bound])).iloc[0]
lower_bound = custom_transform(pd.DataFrame([lower_bound])).iloc[0]
# delta = custom_transform(pd.DataFrame([delta])).iloc[0]

# print(upper_bound)
# print(lower_bound)
# print(delta)

# solver.add(km_driven_z3 >= lower_bound['Kilometers_Driven'], km_driven_z3 <= upper_bound['Kilometers_Driven'])
# solver.add(mileage_z3 >= lower_bound['Mileage'], mileage_z3 <= upper_bound['Mileage'])
# solver.add(engine_z3 >= lower_bound['Engine'], engine_z3 <= upper_bound['Engine'])
# solver.add(power_z3 >= lower_bound['Power'], power_z3 <= upper_bound['Power'])
# solver.add(age_of_car_z3 >= lower_bound['Age_of_Car'], age_of_car_z3 <= upper_bound['Age_of_Car'])  # Age constraint
# solver.add(age_mileage_z3 >= lower_bound['Age_of_Car'] * lower_bound['Kilometers_Driven'], age_mileage_z3 <= upper_bound['Age_of_Car'] * upper_bound['Kilometers_Driven'])
# solver.add(seats_z3 >= lower_bound['Seats'], seats_z3 <= upper_bound['Seats'])
# solver.add(Or(fuel_type_z3 == 0, fuel_type_z3 == 1))  # Fuel type: Diesel or Petrol
# solver.add(Or(transmission_z3 == 0, transmission_z3 == 1))  # Transmission: Manual or Automatic
# solver.add(Or(owner_type_z3 == 1, owner_type_z3 == 2, owner_type_z3 == 3, owner_type_z3 == 4))  # Owner type: 1 to 4
# solver.add(prediction <= lower_bound['Price'])  # Predicted price must be within a valid range


solver.add(km_driven_z3 >= lower_bound['Kilometers_Driven'], km_driven_z3 <= upper_bound['Kilometers_Driven'])
solver.add(mileage_z3 >= lower_bound['Mileage'], mileage_z3 <= upper_bound['Mileage'])
solver.add(engine_z3 >= lower_bound['Engine'],  engine_z3 <= upper_bound['Engine'])
solver.add(power_z3 >= lower_bound['Power'], power_z3 <= upper_bound['Power'])
solver.add(age_of_car_z3 >= lower_bound['Age_of_Car'], age_of_car_z3 <= upper_bound['Age_of_Car'])  # Age constraint
solver.add(seats_z3 >= lower_bound['Seats'], seats_z3 <= upper_bound['Seats'])
solver.add(Or(fuel_type_z3 == 0, fuel_type_z3 == 1))  # Fuel type: Diesel or Petrol
solver.add(Or(transmission_z3 == 0, transmission_z3 == 1))  # Transmission: Manual or Automatic
solver.add(Or(owner_type_z3 == 1, owner_type_z3 == 2, owner_type_z3 == 3, owner_type_z3 == 4))  # Owner type: 1 to 4
solver.add(prediction <= lower_bound['Price'])  # Predicted price must be within a valid range

while (solver.check() == sat) and (iterations <= max_iterations):
    # print(iterations)
    model = solver.model()
    km_val = float(model[km_driven_z3].as_decimal(10).rstrip('?'))
    fuel_val = float(model[fuel_type_z3].as_decimal(10).rstrip('?'))
    transmission_val = float(model[transmission_z3].as_decimal(10).rstrip('?'))
    owner_val = float(model[owner_type_z3].as_decimal(10).rstrip('?'))
    mileage_val = float(model[mileage_z3].as_decimal(10).rstrip('?'))
    engine_val = float(model[engine_z3].as_decimal(10).rstrip('?'))
    power_val = float(model[power_z3].as_decimal(10).rstrip('?'))
    seats_val = float(model[seats_z3].as_decimal(10).rstrip('?'))
    age_car_val = float(model[age_of_car_z3].as_decimal(10).rstrip('?'))
    pr_val, solution = z3_to_solution(km_val, fuel_val, transmission_val, owner_val, mileage_val, engine_val, power_val, seats_val, age_car_val)
    display(solution)
    existing_solutions = append_as_unique_ints(existing_solutions, solution)
    # existing_solutions = pd.concat([existing_solutions, solution], ignore_index=True)

    solver.add(Or(
        z3_abs(km_driven_z3 - km_val) > delta['Kilometers_Driven'],
        z3_abs(fuel_type_z3 - fuel_val) > delta['Fuel_Type'],
        z3_abs(transmission_z3 - transmission_val) > delta['Transmission'],
        z3_abs(owner_type_z3 - owner_val) > delta['Owner_Type'],
        z3_abs(mileage_z3 - mileage_val) > delta['Mileage'],
        z3_abs(engine_z3 - engine_val) > delta['Engine'],
        z3_abs(power_z3 - power_val) > delta['Power'],
        z3_abs(age_of_car_z3 - age_car_val) > delta['Age_of_Car'],
        z3_abs(prediction - pr_val) > delta['Price']
    ))

    if solver.check() != sat:
        break
    iterations += 1

# Output
print(f"Total iterations: {iterations}")
print("Solver state:", solver.check())



# In[229]:


pd.options.display.max_rows = 100
display(existing_solutions)


# In[197]:


display(existing_solutions)
existing_solutions.to_csv('existing_sol_cars.csv')


# In[198]:


new_points = pd.DataFrame(existing_solutions)
lamb = 1
new_points = new_points.rename(columns={'Predicted_Price': 'Price'})
# display(new_points)
new_points['Price'] = (1 - lamb) * new_points['Price'] + lamb * min_reasonable_price
display(new_points)


# In[199]:

# Assuming new_points is your DataFrame with features and a price column
# Exclude the price column for clustering
features = new_points.drop(columns=['Price'])

# Perform HDBSCAN clustering
clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=1)
new_points['cluster'] = clusterer.fit_predict(features)

# Sample points from each cluster
sampled_points_list = []

# Define the number of points to sample from each cluster
points_per_cluster = 10

for cluster in new_points['cluster'].unique():
    cluster_points = new_points[new_points['cluster'] == cluster]
    # Sample a fixed number of points from each cluster or all points if fewer than desired
    sample_size = min(points_per_cluster, len(cluster_points))
    sampled_points_list.append(cluster_points.sample(n=sample_size, random_state=42))

# Concatenate the sampled points into a single DataFrame
sampled_points = pd.concat(sampled_points_list, ignore_index=True)

# Drop the cluster column
sampled_points = sampled_points.drop(columns=['cluster'])
print(sampled_points.head())
# Save the sampled points to a new DataFrame
sampled_points.to_csv('sampled_points_cars.csv', index=False)

# Visualize the clusters
# If you have more than two features, use PCA to reduce to two dimensions
pca = PCA(n_components=2)
features_2d = pca.fit_transform(features)

# Create a scatter plot
plt.figure(figsize=(10, 6))
scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=new_points['cluster'], cmap='viridis', alpha=0.5)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Clusters Visualization with HDBSCAN')
plt.legend(*scatter.legend_elements(), title="Clusters")
plt.show()


# In[200]:


# sampled_points = preprocess(sampled_points)
display(sampled_points.head())


# In[201]:


sampled_points_std = custom_transform(sampled_points, fit=False)

# Split the data into features and target
X_sampled_points = sampled_points_std.drop('Price', axis=1)  # Features
y_sampled_points = sampled_points_std['Price']  # Target variable (Price)

# Split the dataset into training and testing sets
X_train_sampled_points, X_test_sampled_points, y_train_sampled_points, y_test_sampled_points = train_test_split(
    X_sampled_points, y_sampled_points, test_size=0.3, random_state=0)


# In[202]:


# Combine the training datasets
X_train_combined = pd.concat([X_train_cars, X_train_sampled_points], ignore_index=True)
y_train_combined = pd.concat([y_train_cars, y_train_sampled_points], ignore_index=True)

# Combine the testing datasets
X_test_combined = pd.concat([X_test_cars, X_test_sampled_points], ignore_index=True)
y_test_combined = pd.concat([y_test_cars, y_test_sampled_points], ignore_index=True)

# Check the result
print(f"Combined Training Set Shape: {X_train_combined.shape}")
print(f"Combined Testing Set Shape: {X_test_combined.shape}")

# Assign weights
weights_train = np.ones(len(X_train_combined))
weights_train[-len(X_train_sampled_points):] = 1000  # Assign higher weight to the sampled points


# In[203]:


# Initialize the model
weighted_reg_model = LinearRegression()

# Train the model with weights
weighted_reg_model.fit(X_train_combined, y_train_combined, sample_weight=weights_train)

# Make predictions on the test set
y_pred_combined = weighted_reg_model.predict(X_test_combined)

# Calculate performance metrics
rmse_combined = mean_squared_error(y_test_combined, y_pred_combined, squared=False)
r2_combined = r2_score(y_test_combined, y_pred_combined)

# Print out the RMSE and R-squared values
print(f"Weighted Model - RMSE: {rmse_combined}")
print(f"Weighted Model - R² Score: {r2_combined}")


# In[133]:


# Define the features and coefficients
features = [
    "Kilometers_Driven", "Fuel_Type", "Transmission", "Owner_Type",
    "Mileage", "Engine", "Power", "Seats", "Age_of_Car", "Age_Mileage"
]
coefficients = weighted_reg_model.coef_

# Combine features and coefficients into a dictionary
feature_coeff_dict = dict(zip(features, coefficients))

# Sort the features by the absolute value of their coefficients
sorted_features = sorted(feature_coeff_dict.items(), key=lambda x: abs(x[1]), reverse=True)

# Print the sorted feature-importance pairs
print("Feature Coefficients:")
for feature, coef in sorted_features:
    print(f"{feature}: {coef:.4f}")


# In[134]:


y_pred_weighted_sampled_points = weighted_reg_model.predict(X_test_sampled_points)

# Calculate performance metrics for the sampled points in the test set
rmse_sampled_points = mean_squared_error(y_test_sampled_points, y_pred_weighted_sampled_points, squared=False)
r2_sampled_points = r2_score(y_test_sampled_points, y_pred_weighted_sampled_points)

# Print out the RMSE and R-squared values for the sampled points in the test set
print(f"Performance of Auto-Corrected Model on Sampled Points - RMSE: {rmse_sampled_points}")
print(f"Performance of Auto-Corrected Model on Sampled Points - R² Score: {r2_sampled_points}")


# In[135]:


y_pred_normal_sampled_points = linear_reg_model.predict(X_test_sampled_points)

# Calculate performance metrics for the sampled points in the test set
rmse_normal_sampled_points = mean_squared_error(y_test_sampled_points, y_pred_normal_sampled_points, squared=False)
r2_normal_sampled_points = r2_score(y_test_sampled_points, y_pred_normal_sampled_points)

# Print out the RMSE and R-squared values for the sampled points in the test set
print(f"Performance of Normal Model on Sampled Points - RMSE: {rmse_normal_sampled_points}")
print(f"Performance of Normal Model on Sampled Points - R² Score: {r2_normal_sampled_points}")


# In[136]:

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test_cars, y_pred_cars, alpha=0.5)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual Prices vs Predicted Prices - Normal Model')
plt.plot([y_test_cars.min(), y_test_cars.max()], [y_test_cars.min(), y_test_cars.max()], 'k--', lw=2)
plt.show()


# In[137]:


# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test_combined, y_pred_combined, alpha=0.5)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual Prices vs Predicted Prices - Corrected Model')
plt.plot([y_test_combined.min(), y_test_combined.max()], [y_test_combined.min(), y_test_combined.max()], 'k--', lw=2)
plt.show()


# In[138]:


# Create a figure with two subplots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

# Scatter plot for the normal model
axes[0].scatter(y_test_cars, y_pred_cars, alpha=0.5)
axes[0].set_xlabel('Actual Prices')
axes[0].set_ylabel('Predicted Prices')
axes[0].set_title('Actual Prices vs Predicted Prices - Normal Model')
axes[0].plot([y_test_cars.min(), y_test_cars.max()], [y_test_cars.min(), y_test_cars.max()], 'k--', lw=2)

# Scatter plot for the corrected model
axes[1].scatter(y_test_combined, y_pred_combined, alpha=0.5)
axes[1].set_xlabel('Actual Prices')
axes[1].set_ylabel('Predicted Prices')
axes[1].set_title('Actual Prices vs Predicted Prices - Corrected Model')
axes[1].plot([y_test_combined.min(), y_test_combined.max()], [y_test_combined.min(), y_test_combined.max()], 'k--', lw=2)

# Adjust layout
plt.tight_layout()
plt.show()


# In[139]:


# Create a figure with two subplots for the sampled points
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

# print(type(y_pred_normal_sampled_points))

# Scatter plot for the normal model on sampled points
axes[0].scatter(y_test_sampled_points, y_pred_normal_sampled_points, alpha=0.5)
axes[0].set_xlabel('Actual Prices')
axes[0].set_ylabel('Predicted Prices')
axes[0].set_title('Actual Prices vs Predicted Prices - Normal Model on Sampled Points')
axes[0].plot([y_test_sampled_points.min(), y_test_sampled_points.max()], [y_test_sampled_points.min(), y_test_sampled_points.max()], 'k--', lw=2)

# Scatter plot for the corrected model on sampled points
axes[1].scatter(y_test_sampled_points, y_pred_weighted_sampled_points, alpha=0.5)
axes[1].set_xlabel('Actual Prices')
axes[1].set_ylabel('Predicted Prices')
axes[1].set_title('Actual Prices vs Predicted Prices - Corrected Model on Sampled Points')
axes[1].plot([y_test_sampled_points.min(), y_test_sampled_points.max()], [y_test_sampled_points.min(), y_test_sampled_points.max()], 'k--', lw=2)

# Adjust layout
plt.tight_layout()
plt.show()


# In[140]:



# Create figure
fig = go.Figure()

# Fixed x-positions for the two models
x_normal = 0
x_corrected = 1

# Add scatter plots with enhanced styling
fig.add_trace(
    go.Scatter(
        x=[x_normal]*len(y_pred_normal_sampled_points),
        y=y_pred_normal_sampled_points,
        mode='markers',
        name='Normal Model',
        marker=dict(
            size=10,
            color='royalblue',
            symbol='circle',
            line=dict(color='darkblue', width=1)
        )
    )
)

fig.add_trace(
    go.Scatter(
        x=[x_corrected]*len(y_pred_weighted_sampled_points),
        y=y_pred_weighted_sampled_points,
        mode='markers',
        name='Corrected Model',
        marker=dict(
            size=10,
            color='mediumseagreen',
            symbol='circle',
            line=dict(color='darkgreen', width=1)
        )
    )
)

# Add arrows between corresponding points with gradient color
for i in range(len(y_pred_normal_sampled_points)):
    fig.add_shape(
        type="path",
        path=f"M {x_normal},{y_pred_normal_sampled_points[i]} L {x_corrected},{y_pred_weighted_sampled_points[i]}",
        line=dict(
            color="rgba(255,0,0,0.3)",
            width=2,
        ),
        layer='below'
    )

# Update layout with better styling and legend at bottom right
fig.update_layout(
    title=dict(
        text='Prediction Shifts For Corrective Points: Normal → Corrected Model',
        x=0.5,
        font=dict(size=20)
    ),
    xaxis=dict(
        ticktext=['Normal Model', 'Corrected Model'],
        tickvals=[x_normal, x_corrected],
        title='Model Type',
        range=[-0.2, 1.2],
        showgrid=False
    ),
    yaxis=dict(
        title='Predicted Prices',
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray'
    ),
    height=700,
    width=900,
    showlegend=True,
    plot_bgcolor='white',
    paper_bgcolor='white',
    legend=dict(
        yanchor="bottom",
        y=0.01,
        xanchor="right",
        x=0.99
    )
)

fig.show()


# In[141]:

# Sample 100 random indices from test dataset
sample_indices = np.random.choice(len(y_test_combined), size=100, replace=False)

# Get corresponding actual values and predictions
y_test_sample = y_test_combined.iloc[sample_indices]
X_test_sample = X_test_combined.iloc[sample_indices]

# Get predictions from both models
y_pred_normal = pd.Series(linear_reg_model.predict(X_test_sample))
y_pred_corrected = pd.Series(weighted_reg_model.predict(X_test_sample))

# Create figure with subplots
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))

# Color scheme
colors = ['#4B0082', '#0000FF', '#1E90FF']  # Dark to light blue

# Scatter plot for normal model
axes[0].scatter(y_test_sample, y_pred_normal, alpha=0.6, c=colors[1])
axes[0].set_xlabel('Actual Prices')
axes[0].set_ylabel('Predicted Prices')
axes[0].set_title('Normal Model Predictions', pad=15)
axes[0].plot([y_test_sample.min(), y_test_sample.max()], 
             [y_test_sample.min(), y_test_sample.max()], 'k--', lw=1)

# Scatter plot for corrected model
axes[1].scatter(y_test_sample, y_pred_corrected, alpha=0.6, c=colors[2])
axes[1].set_xlabel('Actual Prices')
axes[1].set_ylabel('Predicted Prices')
axes[1].set_title('Corrected Model Predictions', pad=15)
axes[1].plot([y_test_sample.min(), y_test_sample.max()],
             [y_test_sample.min(), y_test_sample.max()], 'k--', lw=1)

# Arrow plot showing shifts
axes[2].set_xlabel('Actual Prices')
axes[2].set_ylabel('Predicted Prices')
axes[2].set_title('Prediction Shifts', pad=15)
axes[2].plot([y_test_sample.min(), y_test_sample.max()],
             [y_test_sample.min(), y_test_sample.max()], 'k--', lw=1)

# Calculate shift magnitudes for coloring
shifts = abs(y_pred_corrected - y_pred_normal)
norm = plt.Normalize(shifts.min(), shifts.max())
cmap = LinearSegmentedColormap.from_list("", ["lightblue", "blue", "darkblue"])

# Add arrows with gradient colors
for i in range(len(y_test_sample)):
    color = cmap(norm(shifts.iloc[i]))
    axes[2].arrow(y_test_sample.iloc[i], y_pred_normal.iloc[i],
                 0, y_pred_corrected.iloc[i] - y_pred_normal.iloc[i],
                 head_width=0.02, head_length=0.05, 
                 fc=color, ec=color, alpha=0.6,
                 length_includes_head=True)

# Style improvements
for ax in axes:
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()

