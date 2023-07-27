# import pandas as pd
# from flask import Flask, render_template, request

# app = Flask(__name__)

# # Read data from CSV and store it in a pandas DataFrame
# data = pd.read_csv('data.csv')

# # Example regression model (you need to train your own model)
# def regression_model(question):
#     # Search for the question (Equation) in the data and return the corresponding answer (Solution)
#     answer = data[data['Equation'] == question]['Solution'].values
#     if len(answer) > 0:
#         return answer[0]
#     # Return 'Not Found' if the question is not in the data
#     return 'Not Found'

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/qa', methods=['POST'])
# def question_answering():
#     if request.method == 'POST':
#         question = request.form['question']
#         answer = regression_model(question)  # Use the regression model to generate the answer
#         return render_template('index.html', question=question, answer=answer, data=data)

# if __name__ == "__main__":
#     app.run(debug=True)



import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Read data from CSV and store it in a pandas DataFrame
data = pd.read_csv('data.csv')

# Example regression model (you need to train your own model)
def regression_model(question):
    # Search for the question (Equation) in the data and return the corresponding answer (Solution) and steps
    result = data[data['Question'] == question]
    if not result.empty:
        answer = result['Answer'].values[0]
        steps = [step for step in result.iloc[0, 1:5] if pd.notna(step)]
        return answer, steps
    # Return 'Not Found' and recommendation if the question is not in the data
    recommendation = 'Recommended source: [Add your recommended source here]'
    return 'Not Found', [recommendation]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/qa', methods=['POST'])
def question_answering():
    if request.method == 'POST':
        question = request.form['question']
        answer, steps = regression_model(question)  # Use the regression model to generate the answer and steps
        return render_template('index.html', question=question, answer=answer, steps=steps)

if __name__ == "__main__":
    app.run(debug=True)
