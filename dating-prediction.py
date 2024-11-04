import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Initialize random seed and sample data
np.random.seed(42)
n_samples = 1000

# Create feature data
data = {
    'อายุ': np.random.randint(18, 50, n_samples),
    'ความสูง': np.random.randint(150, 190, n_samples),
    'การศึกษา': np.random.choice(['มัธยม', 'ปริญญาตรี', 'ปริญญาโท', 'ปริญญาเอก'], n_samples),
    'อาชีพ': np.random.choice(['พนักงานบริษัท', 'ธุรกิจส่วนตัว', 'ข้าราชการ', 'แพทย์', 'วิศวกร'], n_samples),
    'งานอดิเรก': np.random.choice(['กีฬา', 'ดนตรี', 'ท่องเที่ยว', 'อ่านหนังสือ', 'ทำอาหาร'], n_samples),
    'รายได้': np.random.randint(15000, 150000, n_samples),
    'สถานะ': np.random.choice(['โสด', 'หย่า', 'ม่าย'], n_samples)
}

# Generate match results based on certain criteria
def create_match_result(row):
    score = 0
    if row['การศึกษา'] in ['ปริญญาโท', 'ปริญญาเอก']:
        score += 0.3
    if row['รายได้'] > 50000:
        score += 0.2
    if row['อาชีพ'] in ['แพทย์', 'วิศวกร']:
        score += 0.2
    if row['สถานะ'] == 'โสด':
        score += 0.2
    if row['งานอดิเรก'] in ['ท่องเที่ยว', 'กีฬา']:
        score += 0.1
    
    return 1 if score > 0.5 else 0

# Prepare the DataFrame
df = pd.DataFrame(data)
df['match'] = df.apply(create_match_result, axis=1)

# Encode categorical variables
le = LabelEncoder()
categorical_columns = ['การศึกษา', 'อาชีพ', 'งานอดิเรก', 'สถานะ']
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

# Split data into training and testing sets
X = df.drop('match', axis=1)
y = df['match']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
dt_model = DecisionTreeClassifier(random_state=42, max_depth=5)
dt_model.fit(X_train, y_train)

# Test model accuracy
y_pred = dt_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Display model accuracy
st.write(f"### ความแม่นยำของโมเดล: {accuracy * 100:.2f}%")
st.write("### รายงานการจำแนกประเภท:")
st.text(report)

# Function to predict match for new input
def predict_match(model, age, height, education, occupation, hobby, income, status):
    input_data = pd.DataFrame({
        'อายุ': [age],
        'ความสูง': [height],
        'การศึกษา': [education],
        'อาชีพ': [occupation],
        'งานอดิเรก': [hobby],
        'รายได้': [income],
        'สถานะ': [status]
    })
    
    for col in categorical_columns:
        input_data[col] = le.transform(input_data[col])
    
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)
    
    return prediction[0], probability[0]

# Interactive inputs for new prediction
st.write("## ทำนายคู่แมทช์ใหม่")
age = st.slider('อายุ', 18, 50, 28)
height = st.slider('ความสูง', 150, 190, 175)
education = st.selectbox('การศึกษา', ['มัธยม', 'ปริญญาตรี', 'ปริญญาโท', 'ปริญญาเอก'])
occupation = st.selectbox('อาชีพ', ['พนักงานบริษัท', 'ธุรกิจส่วนตัว', 'ข้าราชการ', 'แพทย์', 'วิศวกร'])
hobby = st.selectbox('งานอดิเรก', ['กีฬา', 'ดนตรี', 'ท่องเที่ยว', 'อ่านหนังสือ', 'ทำอาหาร'])
income = st.number_input('รายได้', min_value=15000, max_value=150000, value=80000, step=5000)
status = st.selectbox('สถานะ', ['โสด', 'หย่า', 'ม่าย'])

# Prediction
if st.button("ทำนาย"):
    result, prob = predict_match(dt_model, age, height, education, occupation, hobby, income, status)
    st.write("### ผลการทำนาย:")
    st.write(f"{'เข้ากันได้' if result == 1 else 'ไม่เข้ากัน'}")
    st.write(f"### ความน่าจะเป็น: {max(prob) * 100:.2f}%")
