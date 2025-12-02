import gradio as gr

def infer_ui(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked_C, Embarked_Q):
    return predict_survival(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked_C, Embarked_Q)

gr.Interface(
    fn=infer_ui,
    inputs=[
        gr.Dropdown([1, 2, 3], label="Hạng vé (Pclass)"),
        gr.Dropdown(["male", "female"], label="Giới tính"),
        gr.Slider(1, 80, label="Tuổi"),
        gr.Slider(0, 8, label="Số anh chị em đi cùng"),
        gr.Slider(0, 6, label="Số cha mẹ/con đi cùng"),
        gr.Slider(0, 600, label="Giá vé"),
        gr.Checkbox(label="Lên tàu tại Cherbourg (C)"),
        gr.Checkbox(label="Lên tàu tại Queenstown (Q)")
    ],
    outputs="text",
    title="Dự đoán khả năng sống sót trên Titanic"
).launch()

