import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="แอพทำนายความเข้ากันของคู่แมทช์", layout="wide")
st.title("🚀 แอพทำนายความเข้ากันของคู่แมทช์")
st.write("ใช้ Machine Learning ในการทำนายความเข้ากันของคู่แมทช์จากข้อมูลโปรไฟล์")

# สร้างข้อมูลตัวอย่าง
@st.cache_data
def generate_sample_data():
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'อายุ': np.random.randint(18, 50, n_samples),
        'ความสูง': np.random.randint(150, 190, n_samples),
        'การศึกษา': np.random.choice(['มัธยม', 'ปริญญาตรี', 'ปริญญาโท', 'ปริญญาเอก'], n_samples),
        'อาชีพ': np.random.choice(['พนักงานบริษัท', 'ธุรกิจส่วนตัว', 'ข้าราชการ', 'แพทย์', 'วิศวกร'], n_samples),
        'งานอดิเรก': np.random.choice(['กีฬา', 'ดนตรี', 'ท่องเที่ยว', 'อ่านหนังสือ', 'ทำอาหาร'], n_samples),
        'รายได้': np.random.randint(15000, 150000, n_samples),
        'สถานะ': np.random.choice(['โสด', 'หย่า', 'ม่าย'], n_samples)
    }
    return pd.DataFrame(data)

# ฟังก์ชันสร้างผลลัพธ์การแมทช์
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

# เตรียมข้อมูลและเทรนโมเดล
@st.cache_resource
def train_model():
    df = generate_sample_data()
    df['match'] = df.apply(create_match_result, axis=1)
    
    le = LabelEncoder()
    categorical_columns = ['การศึกษา', 'อาชีพ', 'งานอดิเรก', 'สถานะ']
    df_encoded = df.copy()
    
    for col in categorical_columns:
        df_encoded[col] = le.fit_transform(df_encoded[col])
    
    X = df_encoded.drop('match', axis=1)
    y = df_encoded['match']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = DecisionTreeClassifier(random_state=42, max_depth=5)
    model.fit(X_train, y_train)
    
    return model, le, categorical_columns

# โหลดโมเดลและข้อมูลที่จำเป็น
model, le, categorical_columns = train_model()

# สร้าง sidebar สำหรับใส่ข้อมูล
st.sidebar.header("📝 กรอกข้อมูลโปรไฟล์")

# รับข้อมูลจากผู้ใช้
age = st.sidebar.slider("อายุ", 18, 50, 25)
height = st.sidebar.slider("ความสูง (ซม.)", 150, 190, 170)
education = st.sidebar.selectbox("การศึกษา", ['มัธยม', 'ปริญญาตรี', 'ปริญญาโท', 'ปริญญาเอก'])
occupation = st.sidebar.selectbox("อาชีพ", ['พนักงานบริษัท', 'ธุรกิจส่วนตัว', 'ข้าราชการ', 'แพทย์', 'วิศวกร'])
hobby = st.sidebar.selectbox("งานอดิเรก", ['กีฬา', 'ดนตรี', 'ท่องเที่ยว', 'อ่านหนังสือ', 'ทำอาหาร'])
income = st.sidebar.slider("รายได้ (บาท/เดือน)", 15000, 150000, 30000, step=1000)
status = st.sidebar.selectbox("สถานะ", ['โสด', 'หย่า', 'ม่าย'])

# สร้างฟังก์ชันทำนาย
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
    
    input_encoded = input_data.copy()
    for col in categorical_columns:
        input_encoded[col] = le.fit_transform(input_encoded[col])
    
    prediction = model.predict(input_encoded)
    probability = model.predict_proba(input_encoded)
    
    return prediction[0], probability[0]

# ปุ่มทำนาย
if st.sidebar.button("ทำนายความเข้ากัน"):
    result, prob = predict_match(
        model,
        age,
        height,
        education,
        occupation,
        hobby,
        income,
        status
    )
    
    # แสดงผลลัพธ์ในรูปแบบที่สวยงาม
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 ผลการทำนาย")
        if result == 1:
            st.success("✨ เข้ากันได้!")
            st.balloons()
        else:
            st.error("❌ ไม่เข้ากัน")
            
    with col2:
        st.subheader("🎯 ความน่าจะเป็น")
        st.progress(max(prob))
        st.write(f"ความน่าจะเป็น: {max(prob) * 100:.2f}%")
    
    # แสดงข้อมูลโปรไฟล์
    st.subheader("👤 ข้อมูลโปรไฟล์ที่ใช้ในการทำนาย")
    profile_data = {
        "หัวข้อ": ["อายุ", "ความสูง", "การศึกษา", "อาชีพ", "งานอดิเรก", "รายได้", "สถานะ"],
        "ค่า": [age, f"{height} ซม.", education, occupation, hobby, f"{income:,} บาท", status]
    }
    st.table(pd.DataFrame(profile_data))

# แสดงข้อมูลเพิ่มเติม
with st.expander("ℹ️ เกี่ยวกับแอพนี้"):
    st.write("""
    แอพนี้ใช้ Machine Learning (Decision Tree) ในการทำนายความเข้ากันของคู่แมทช์ 
    โดยพิจารณาจากปัจจัยต่างๆ เช่น อายุ การศึกษา อาชีพ และรายได้
    
    **วิธีใช้งาน:**
    1. กรอกข้อมูลโปรไฟล์ในแถบด้านซ้าย
    2. กดปุ่ม "ทำนายความเข้ากัน"
    3. ระบบจะแสดงผลการทำนายพร้อมความน่าจะเป็น
    
    **หมายเหตุ:** ผลการทำนายนี้เป็นเพียงการคาดการณ์เท่านั้น
    """)
