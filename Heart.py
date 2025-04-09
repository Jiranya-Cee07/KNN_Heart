from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.title('การทำนายโรคหัวใจด้วย K-Nearest Neighbor')
#st.image("./img/cat.jpg")
col1, col2, col3 = st.columns(3)


with col1:
   st.header("จิรัญญา")
   st.image("./img/me.jpg")

with col2:
   st.header("ไม่เป็นโรคหัวใจ")
   st.image("./img/H.jpg")

with col3:
   st.header("เป็นโรคหัวใจ")
   st.image("./img/H1.jpg")

html_7 = """
<div style="background-color:#c5f18a;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h3>ข้อมูลสำหรับทำนาย</h3></center>
</div>
"""
st.markdown(html_7, unsafe_allow_html=True)
st.markdown("")
st.markdown("")

st.subheader("ข้อมูลส่วนแรก 5 แถว")
dt = pd.read_csv("./data/Heart3.csv")
st.write(dt.head(5))
st.subheader("ข้อมูลส่วนสุดท้าย 5 แถว")
st.write(dt.tail(5))

# สถิติพื้นฐาน
st.subheader("📈 สถิติพื้นฐานของข้อมูล")
st.write(dt.describe())

# การเลือกแสดงกราฟตามฟีเจอร์
st.subheader("📌 เลือกฟีเจอร์เพื่อดูการกระจายข้อมูล")
feature = st.selectbox("เลือกฟีเจอร์", dt.columns[:-1])

# วาดกราฟ boxplot
st.write(f"### 🎯 Boxplot: {feature} แยกตามชนิดของเป็นโรคหัวใจ/ไม่เป็นโรคหัวใจ")
fig, ax = plt.subplots()
sns.boxplot(data=dt, x='HeartDisease', y=feature, ax=ax)
st.pyplot(fig)

# วาด pairplot
if st.checkbox("แสดง Pairplot (ใช้เวลาประมวลผลเล็กน้อย)"):
    st.write("### Pairplot: การกระจายของข้อมูลทั้งหมด")
    fig2 = sns.pairplot(dt, hue='HeartDisease')
    st.pyplot(fig2)


html_8 = """
<div style="background-color:#6BD5DA;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>ทำนายข้อมูล</h5></center>
</div>
"""
st.markdown(html_8, unsafe_allow_html=True)
st.markdown("")

pt_age = st.number_input("กรุณาเลือกข้อมูล Age")
pt_sex = st.number_input("กรุณาเลือกข้อมูล Sex")
sp_ChestPainType = st.number_input("กรุณาเลือกข้อมูล ChestPainType")
sp_RestingBP = st.number_input("กรุณาเลือกข้อมูล RestingBP")
pt_Cholesterol = st.number_input("กรุณาเลือกข้อมูล Cholesterol")
pt_FastingBS = st.number_input("กรุณาเลือกข้อมูล FastingBS")
sp_RestingECG = st.number_input("กรุณาเลือกข้อมูล RestingECG")
sp_MaxHR = st.number_input("กรุณาเลือกข้อมูล MaxHR")
sp_ExerciseAngina = st.number_input("กรุณาเลือกข้อมูล ExerciseAngina")
sp_Oldpeak = st.number_input("กรุณาเลือกข้อมูล Oldpeak")
sp_ST_Slope = st.number_input("กรุณาเลือกข้อมูล ST_Slope")

if st.button("ทำนายผล"):
    #st.write("ทำนาย")
   dt = pd.read_csv("./data/Heart3.csv") 
   X = dt.drop('HeartDisease', axis=1)
   y = dt.HeartDisease  

   Knn_model = KNeighborsClassifier(n_neighbors=3)
   Knn_model.fit(X, y)  
    
   x_input = np.array([[pt_age, pt_sex, sp_ChestPainType, sp_RestingBP, pt_Cholesterol, pt_FastingBS, sp_RestingECG, sp_MaxHR, sp_ExerciseAngina,
                        sp_Oldpeak, sp_ST_Slope]])
   st.write(Knn_model.predict(x_input))
   
   out=Knn_model.predict(x_input)

   if out[0] == '1':
    st.write("เป็นโรคหัวใจ")
    st.image("./img/H1.jpg")
   else:
    st.write("ไม่เป็นโรคหัวใจ")
    st.image("./img/H.jpg")
else:
    st.write("ไม่ทำนาย")
