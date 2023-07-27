import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering, AdamW
from torch.utils.data import DataLoader, Dataset

class MathQADataset(Dataset):
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
    # Load your data from CSV
    data = pd.read_csv('math_qa_dataset.csv')

    # Clean the data: Remove rows with missing values and duplicates
    data = data.dropna()
    data = data.drop_duplicates()

    # Additional questions and answers
    additional_data = {
        'question': [
            "What is 2 + 2?",
            "Solve for x: 3x + 5 = 17",
            "What is the square root of 49?",
            "Evaluate: 2 * (3 + 4)",
            "Find the value of y: 4y - 10 = 30",
            "Calculate: 12 ÷ 2 + 3",
            "Simplify: 3(2x + 5) - 2x",
            "What is 4 squared?",
            "Solve for x: 3x - 7 = 14",
            "Find the area of a rectangle with length 8 and width 5",
            # Add 10 more questions here...
            "Evaluate: 5 * (6 + 7)",
            "What is 10% of 2000?",
            "Calculate: 3^4",
            "Solve for x: 5x + 8 = 33",
            "Find the volume of a cube with side length 6",
            "What is 25% of 80?",
            "Simplify: 2(3x + 4) - 5x",
            "Find the perimeter of a square with sides of length 5",
            "Evaluate: 4 + 3 * 2",
            "What is the area of a triangle with base 10 and height 8?",
        ],
        'answer': [
            "4",
            "4",
            "7",
            "14",
            "10",
            "9",
            "4x + 15",
            "16",
            "7",
            "40",
            # Add answers for the 10 additional questions here...
            "65",
            "200",
            "81",
            "5",
            "216",
            "20",
            "6x + 3",
            "20",
            "10",
            "40",
        ]
    }

    # Check if the lengths are different
    if len(additional_data['question']) > len(additional_data['answer']):
        # Assume the missing answer corresponds to the last question
        num_missing_answers = len(additional_data['question']) - len(additional_data['answer'])
        additional_data['answer'] += [""] * num_missing_answers

    # Create the DataFrame
    additional_df = pd.DataFrame(additional_data)
    data = pd.concat([data, additional_df])

    # Reset DataFrame index to consecutive integers
    data.reset_index(drop=True, inplace=True)

    # Tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased")

    # Preprocess the dataset
    train_dataset = MathQADataset(data, tokenizer, max_length=256)  # Change max_length as needed

    # Model
    model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-cased")

    # Fine-tune the model
    optimizer = AdamW(model.parameters(), lr=5e-5)
    model.train()

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    for epoch in range(3):  # Change the number of epochs as needed
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].squeeze(1)
            attention_mask = batch['attention_mask'].squeeze(1)
            start_positions = batch['start_positions']
            end_positions = batch['end_positions']

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

    # Save the fine-tuned model
    model.save_pretrained("P2EPAIS_tuned_model")
    tokenizer.save_pretrained("P2EPAIS_tuned_model")

if __name__ == "__main__":
    main()
