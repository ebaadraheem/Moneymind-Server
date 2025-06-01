# Moneymind-Server

🛠️ Installation & Setup
Follow these steps to clone and run the Moneymind Server locally:

1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/ebaadraheem/Moneymind-Server.git
cd Moneymind-Server

3. Create a Virtual Environment
bash
Copy
Edit
python -m venv venv
4. Activate the Virtual Environment
5. 
On Windows:

bash
Copy
Edit
venv\Scripts\activate
On macOS/Linux:

bash
Copy
Edit
source venv/bin/activate

4. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
5. Run the Flask Application
bash
Copy
Edit
python main.py

📁 Project Structure
bash
Copy
Edit
Moneymind-Server/
│
├── main.py              # Main Flask app
├── requirements.txt    # Project dependencies
├── README.md           # This file
├── .env           # Env file
└── .gitignore          # Git ignored files (should include venv/)

#Include the .env file 
GEMINI_API_KEY="Gemini model api key"
GOOGLE_APPLICATION_CREDENTIALS="Path to your firebase file to access database"
CORS_ALLOWED_ORIGINS="Frontend url"
