import pandas as pd
import random
import torch
from sympy import *
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
from torch.optim import AdamW  # Import the AdamW optimizer from torch.optim

# Set seed for reproducibility
random.seed(42)

def clean_equation(eq):
    return eq

def clean_solution(sol):
    sol_str = str(sol)
    # Extract the part between the parentheses (i.e., the solutions)
    sol_str = sol_str.split('(')[1].split(')')[0].strip()
    return sympify(sol_str)

def generate_math_question():
    x, y = symbols('x y')

    # Manually choose a set of equations and their solutions
    equations = [
        (x**2 + 3*y**2 - 4, solve(x**2 + 3*y**2 - 4, x)),
        (2*x**3 + 5*y**3 - 15, solve(2*x**3 + 5*y**3 - 15, x)),
        (x**4 + 6*y**4 - 9, solve(x**4 + 6*y**4 - 9, x)),
        (3*x**3 + 8*y**3 - 6, solve(3*x**3 + 8*y**3 - 6, x)),
        (2*x**2 + 4*y**2 - 8, solve(2*x**2 + 4*y**2 - 8, x)),
    ]
    
    # Randomly choose an equation from the list
    expression, answer = random.choice(equations)

    # Convert answer to a string
    answer_str = ', '.join(map(str, answer))

    return clean_equation(expression), clean_solution(answer_str)

def generate_math_questions(num_questions):
    questions = []
    for _ in range(num_questions):
        question, answer = generate_math_question()
        questions.append((f"What is {question}?", answer))

    return questions

class MathQADataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        question = self.data.loc[index, 'question']
        answer = self.data.loc[index, 'answer']
        encoding = self.tokenizer(question, answer, padding='max_length', truncation=True, return_tensors='pt', max_length=self.max_length)
        input_ids = encoding['input_ids'][0]
        attention_mask = encoding['attention_mask'][0]
        start_positions = torch.tensor([input_ids.tolist().index(102)])  # Index of '[CLS]'
        end_positions = torch.tensor([len(input_ids) - 1])  # Last token (excluding padding) as end position
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'start_positions': start_positions,
            'end_positions': end_positions
        }

def main():
    num_questions = 5000
    questions = generate_math_questions(num_questions)
    data = {'question': [q[0] for q in questions], 'answer': [q[1] for q in questions]}
    df = pd.DataFrame(data)

    # Reset DataFrame index to consecutive integers
    df.reset_index(drop=True, inplace=True)

    # Save the DataFrame to 'data.csv'
    df.to_csv('data.csv', index=False)

    # Replace the deprecated AdamW optimizer with torch.optim.AdamW
    model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-cased')
    optimizer = AdamW(model.parameters())  # Use torch.optim.AdamW

    # Load your data from CSV
    data = pd.read_csv('data.csv')

    # Clean the data: Remove rows with missing values and duplicates
    data = data.dropna()
    data = data.drop_duplicates()

    # Tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased")

    # Preprocess the dataset
    train_dataset = MathQADataset(data, tokenizer, max_length=256)  # Change max_length as needed

    # Data loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)

    # Loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Training loop
    model.train()
    num_epochs = 3  # Change the number of epochs as needed
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            start_positions = batch['start_positions']
            end_positions = batch['end_positions']

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss}")

    # Save the fine-tuned model
    model.save_pretrained("P2EPAIS_tuned_model")
    tokenizer.save_pretrained("P2EPAIS_tuned_model")

if __name__ == "__main__":
    main()
