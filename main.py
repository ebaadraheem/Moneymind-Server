import os
import logging
import functools # For functools.wraps
from datetime import datetime, timezone # For generating timestamps and default titles

from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, auth, firestore

from dotenv import load_dotenv

load_dotenv() # Load .env for local dev

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    handlers=[
        logging.StreamHandler() # Outputs to console
    ]
)

# --- Firestore Constants ---
USER_COLLECTION = "users"
CHAT_SESSIONS_SUBCOLLECTION = "chat_sessions"
MESSAGES_SUBCOLLECTION = "messages"
DEFAULT_CHAT_TITLE_PREFIX = "New Chat -" # Used to check if title needs update

# --- Firebase Admin SDK Initialization ---
db = None
firebase_app = None
try:
    cred = credentials.ApplicationDefault()
    firebase_app = firebase_admin.initialize_app(cred)
    db = firestore.client()
    logging.info("Firebase Admin SDK initialized with Application Default Credentials.")
except Exception as e_default:
    logging.warning(f"Failed to initialize Firebase with Application Default Credentials: {e_default}")
    cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if cred_path:
        logging.info(f"Attempting Firebase initialization with service account key: {cred_path}")
        try:
            cred = credentials.Certificate(cred_path)
            if not firebase_admin._apps: # Check if default app already initialized
                firebase_app = firebase_admin.initialize_app(cred)
            else:
                firebase_app = firebase_admin.get_app() # Get default app if already initialized
            db = firestore.client()
            logging.info(f"Firebase Admin SDK initialized successfully with service account key: {cred_path}.")
        except Exception as e_path:
            logging.error(f"🔴 Firebase Admin SDK initialization failed with service account key {cred_path}: {e_path}", exc_info=True)
    else:
        logging.warning("🔴 GOOGLE_APPLICATION_CREDENTIALS environment variable not set, and Application Default failed.")

if not db:
    logging.critical("🔴🔴🔴 Firestore client (db) could not be initialized. Database operations will fail. 🔴🔴🔴")

# --- Gemini API Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
model = None

if not GEMINI_API_KEY:
    logging.critical("🔴 Gemini API Key (GEMINI_API_KEY) not found. AI features will be disabled.")
else:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        generation_config = {
            "temperature": 0.7, "top_p": 1, "top_k": 1, "max_output_tokens": 2048,
        }
        system_instruction = """You are Moneymind, a highly knowledgeable and professional AI Finance Assistant. Your primary function is to provide clear, accurate, and comprehensive information and answer questions *strictly* related to the broad domain of finance. You should be friendly, patient, and aim to empower users with financial understanding.

        **Your Core Financial Expertise Includes (but is not limited to):**

        1.  **Personal Finance:**
            * Budgeting and Expense Tracking: Strategies, tools, creating budgets.
            * Saving and Investing: Different types of accounts (savings, checking, money market), emergency funds, goal-setting.
            * Debt Management: Credit cards, loans (student, personal, auto), debt consolidation, strategies for paying off debt.
            * Credit Scores & Reports: Understanding credit scores, how they're calculated, improving credit, checking reports.
            * Insurance: Types of insurance (health, life, auto, home, disability), understanding policies, choosing coverage.
            * Retirement Planning: Concepts like 401(k), IRA (Traditional, Roth), pensions, planning for retirement income.
            * Major Purchases: Financial considerations for buying a home (mortgages, down payments, closing costs) or a car.
            * Financial Goal Setting: Short-term and long-term financial planning.

        2.  **Investing & Markets:**
            * Asset Classes: Stocks, bonds, mutual funds, Exchange-Traded Funds (ETFs), real estate, commodities.
            * Investment Strategies: Value investing, growth investing, diversification, asset allocation, risk tolerance.
            * Market Analysis: Understanding market trends, economic indicators (inflation, GDP, unemployment), fundamental and technical analysis concepts.
            * Stock Market: How it works, stock exchanges, indices (e.g., S&P 500, Dow Jones), IPOs.
            * Brokerage Accounts: Types of accounts, how to choose a broker.
            * Financial Instruments: Options, futures, derivatives (explain concepts, risks, and uses).
            * Alternative Investments: Venture capital, private equity, hedge funds (explain concepts).

        3.  **Cryptocurrency & Digital Assets:**
            * Blockchain Technology: Basic principles.
            * Cryptocurrencies: Bitcoin, Ethereum, altcoins (explain what they are, use cases, risks).
            * Decentralized Finance (DeFi): Concepts, lending, borrowing, yield farming (explain at a high level).
            * Centralized Finance (CeFi) in Crypto: Exchanges, custody.
            * Crypto Wallets & Security: Types of wallets, best practices for security.
            * Non-Fungible Tokens (NFTs): What they are, use cases, market dynamics.
            * Initial Coin Offerings (ICOs), IDOs, STOs: Concepts and risks.

        4.  **Business & Corporate Finance:**
            * Financial Statements: Balance sheet, income statement, cash flow statement (how to read and interpret them).
            * Business Valuation: Basic concepts.
            * Entrepreneurship: Funding startups, business loans, financial planning for small businesses.
            * Economics: Micro and macro-economic principles, supply and demand, market structures.
            * Mergers & Acquisitions (M&A): Basic concepts.

        5.  **Money Management & Banking:**
            * Banking Products: Checking accounts, savings accounts, certificates of deposit (CDs).
            * Loans & Mortgages: Types, interest rates, amortization.
            * Interest Rates & Inflation: How they work and their impact.
            * Monetary & Fiscal Policy: Basic understanding of central bank roles and government economic policies.

        6.  **Financial Scams & Security:**
            * Identifying Scams: Phishing, Ponzi schemes, pyramid schemes, pump-and-dump schemes.
            * Prevention Strategies: Protecting personal information, secure online practices.
            * Reporting Mechanisms: General guidance on where to report financial fraud.

        **Your Interaction Guidelines:**

        * **Greetings & Introduction:** When a user initiates a conversation or sends a greeting, you should always respond in a friendly manner and include your name, Moneymind. For example:
            * "Hello! I'm Moneymind, your AI Finance Assistant. How can I help you with your financial questions today?"
            * "Hi there! Moneymind here, ready to assist with your finance-related queries. What's on your mind?"
            * "Welcome! I am Moneymind. Feel free to ask me anything about finance."
        * **Strictly On-Topic:** You *must not* answer any questions or engage in any conversation outside of these financial topics.
        * **Polite Refusal for Off-Topic Queries:** If a user asks a non-finance question (e.g., about politics, sports, personal opinions, general knowledge outside finance), you must politely and firmly state that your expertise is limited to finance-related matters and you cannot assist with that specific query. For example: "My apologies, but as Moneymind, my expertise is limited to finance-related matters and I cannot assist with that specific query." or "That's an interesting question! However, my knowledge as Moneymind is focused on finance. Is there a financial topic I can help you with today?"
        * **No Financial Advice:** You *must not* provide specific financial, investment, legal, or tax advice. Do not tell users what to buy or sell, or whether a specific investment is "good" or "bad" for them.
            * Instead: Provide general information, explain concepts, discuss different perspectives, describe potential risks and benefits, and offer educational content. You can discuss historical performance patterns or typical characteristics of asset classes.
            * Example of what NOT to say: "You should buy XYZ stock."
            * Example of what TO say: "XYZ stock is in the tech sector. Stocks in this sector can be volatile but also offer growth potential. When considering any stock, it's important to look at the company's fundamentals, your own risk tolerance, and diversify your portfolio. Would you like to know more about how to analyze a company's fundamentals?"
        * **Informational & Educational:** Your role is to inform and educate. Discuss all aspects of financial topics, including high-risk areas (like speculative investments or volatile cryptocurrencies), from an objective, informational perspective, always highlighting potential risks.
        * **Data Limitations:** If asked for real-time or live market prices, state that you do not have access to live data feeds but can provide general information about where such data might be found or discuss historical price movements based on your general knowledge.
        * **Professional & Friendly Tone:** Maintain a helpful, patient, and professional demeanor. Use clear and concise language, avoiding overly technical jargon where possible, or explaining it if necessary.
        * **Encourage Learning:** When appropriate, you can guide users towards further learning or suggest related financial topics they might find interesting.
        * **Conciseness and Accuracy:** Strive for factual accuracy based on your training data and provide responses that are as concise as possible while still being comprehensive.

        By adhering to these instructions, you will be a valuable and trusted AI Finance Assistant.
        """
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash-latest",
            generation_config=generation_config,
            system_instruction=system_instruction,
            safety_settings=safety_settings
        )
        logging.info("Gemini Model initialized successfully.")
    except Exception as e:
        logging.error(f"🔴 Error initializing Gemini Model: {e}", exc_info=True)
        model = None

app = Flask(__name__)

# --- CORS Configuration ---
allowed_origins_env = os.getenv("CORS_ALLOWED_ORIGINS")
if allowed_origins_env:
    origins = allowed_origins_env.split(',')
    logging.info(f"CORS: Allowing specific origins: {origins}")
else:
    # Fallback to a default if not set, but strongly recommend setting the environment variable.
    origins = ["http://localhost:5173", "http://localhost:3000"] # Example: Common React dev port
    # The ngrok URL was specific to a temporary session, it's better to use localhost for general dev
    # or the actual deployed frontend URL if known.
    logging.warning(f"CORS: CORS_ALLOWED_ORIGINS environment variable not set. Defaulting to {origins}. Please set this for production environments.")
CORS(
    app, resources={r"/api/*": {"origins": origins}},
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    supports_credentials=True, expose_headers=["Content-Length"], max_age=86400
)

# --- Authentication Middleware (Decorator) ---
def check_auth(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        if request.method == 'OPTIONS':
            response = make_response()
            # CORS headers are typically handled by Flask-CORS,
            # but for OPTIONS, an explicit 200/204 is good.
            response.status_code = 204 # No Content is often used for OPTIONS preflight
            return response

        if not firebase_app:
            logging.error("Firebase app not initialized. Authentication check skipped, returning 503.")
            return jsonify({"error": "Authentication service unavailable"}), 503
        
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            logging.warning("Unauthorized access: Missing or invalid Authorization header.")
            return jsonify({"error": "Unauthorized - Missing or invalid token"}), 401
        
        id_token = auth_header.split('Bearer ')[1]
        try:
            # Pass the initialized Firebase app instance to verify_id_token
            decoded_token = auth.verify_id_token(id_token, app=firebase_app)
            request.user_id = decoded_token['uid']
            # If chat_id is part of the URL, add it to the request context for convenience
            if 'chat_id' in kwargs:
                request.chat_id = kwargs['chat_id']
            logging.debug(f"User authenticated: {request.user_id}")
            return f(*args, **kwargs) # Call the actual decorated function
        except auth.RevokedIdTokenError:
            logging.warning(f"Auth Error: Token revoked for supposed UID {id_token[:10]}...")
            return jsonify({"error": "Unauthorized - Token has been revoked"}), 401
        except auth.UserDisabledError:
            logging.warning(f"Auth Error: User account disabled for supposed UID {id_token[:10]}...")
            return jsonify({"error": "Unauthorized - User account is disabled"}), 401
        except auth.InvalidIdTokenError as e: # Catches expired tokens and other invalid token issues
            logging.warning(f"Auth Error: Invalid ID token ({e}): {id_token[:10]}...")
            return jsonify({"error": "Unauthorized - Invalid ID token"}), 401
        except Exception as e:
            logging.error(f"🔴 Auth Error during token verification: {e}", exc_info=True)
            return jsonify({"error": "Unauthorized - Token verification failed"}), 401
    return wrapper

# --- Gemini Interaction Function ---
def get_finance_response_backend(user_prompt_text: str, chat_history_from_client: list):
    if not model:
        logging.error("Gemini model is not available.")
        # Return the original history if the model isn't available, so the client doesn't lose context.
        return "Sorry, the AI model is not available right now. Please try again later.", chat_history_from_client
    try:
        # Ensure history items are correctly formatted Content objects if needed by the SDK version
        # For recent versions, list of dicts {'role': ..., 'parts': [text]} is standard.
        convo = model.start_chat(history=chat_history_from_client)
        convo.send_message(user_prompt_text)
        # convo.last should be the model's response
        # convo.history includes the input history, the user's last message, and the model's last response.
        return convo.last.text, convo.history
    except Exception as e:
        logging.error(f"🔴 Error getting response from Gemini: {e}", exc_info=True)
        return f"Sorry, I encountered an error processing your request: {str(e)[:100]}...", chat_history_from_client


# --- API Endpoints for Multi-Chat Sessions ---

@app.route('/api/hello', methods=['GET'])
def hello():
    return jsonify(message="Hello from Flask on Firebase Cloud Functions!")


@app.route('/api/chats', methods=['GET'])
@check_auth
def list_chat_sessions():
    """Lists all chat sessions for the authenticated user."""
    if not db:
        return jsonify({"error": "Database service temporarily unavailable"}), 503
    user_id = request.user_id
    try:
        sessions_ref = db.collection(USER_COLLECTION).document(user_id).collection(CHAT_SESSIONS_SUBCOLLECTION)
        # Order by lastUpdatedAt first (desc), then by createdAt (desc) as a secondary sort for sessions.
        sessions_query = sessions_ref.order_by("lastUpdatedAt", direction=firestore.Query.DESCENDING)\
                                     .order_by("createdAt", direction=firestore.Query.DESCENDING)
        sessions_docs = sessions_query.stream()

        sessions = []
        for doc in sessions_docs:
            session_data = doc.to_dict()
            session_data['id'] = doc.id
            # Convert Firestore Timestamps to ISO format strings for JSON serialization
            if 'createdAt' in session_data and isinstance(session_data['createdAt'], datetime):
                session_data['createdAt'] = session_data['createdAt'].isoformat()
            if 'lastUpdatedAt' in session_data and isinstance(session_data['lastUpdatedAt'], datetime):
                session_data['lastUpdatedAt'] = session_data['lastUpdatedAt'].isoformat()
            sessions.append(session_data)

        logging.info(f"Fetched {len(sessions)} chat sessions for user {user_id}")
        return jsonify({"sessions": sessions})
    except Exception as e:
        logging.error(f"🔴 Error fetching chat sessions for user {user_id}: {e}", exc_info=True)
        return jsonify({"error": "Could not fetch chat sessions"}), 500

@app.route('/api/chats', methods=['POST'])
@check_auth
def create_chat_session():
    """Creates a new chat session for the authenticated user."""
    if not db:
        return jsonify({"error": "Database service temporarily unavailable"}), 503
    user_id = request.user_id
    try:
        current_server_time = firestore.SERVER_TIMESTAMP # For DB atomicity
        now_for_display_and_title = datetime.now(timezone.utc) # For immediate response and default title

        default_title = f"{DEFAULT_CHAT_TITLE_PREFIX} {now_for_display_and_title.strftime('%b %d, %H:%M')}"

        new_session_ref = db.collection(USER_COLLECTION).document(user_id).collection(CHAT_SESSIONS_SUBCOLLECTION).document()
        session_data = {
            "title": default_title,
            "createdAt": current_server_time,
            "lastUpdatedAt": current_server_time,
            "userId": user_id # Store userId for potential cross-user admin features or rules.
        }
        new_session_ref.set(session_data)

        # For the response, use the client-generated timestamp for createdAt/lastUpdatedAt
        # as SERVER_TIMESTAMP won't be resolved until commit.
        session_response_data = {
            "id": new_session_ref.id,
            "title": default_title,
            "createdAt": now_for_display_and_title.isoformat(), # ISO format for consistency
            "lastUpdatedAt": now_for_display_and_title.isoformat()
        }

        logging.info(f"Created new chat session {new_session_ref.id} for user {user_id} with title '{default_title}'")
        return jsonify({"session": session_response_data}), 201
    except Exception as e:
        logging.error(f"🔴 Error creating new chat session for user {user_id}: {e}", exc_info=True)
        return jsonify({"error": "Could not create new chat session"}), 500


@app.route('/api/chats/<string:chat_id>/history', methods=['GET'])
@check_auth
def get_chat_history(chat_id: str):
    """Fetches message history for a specific chat session."""
    if not db:
        return jsonify({"error": "Database service temporarily unavailable"}), 503
    user_id = request.user_id
    try:
        messages_ref = db.collection(USER_COLLECTION).document(user_id).collection(CHAT_SESSIONS_SUBCOLLECTION).document(chat_id).collection(MESSAGES_SUBCOLLECTION)
        
        # MODIFIED QUERY: Added order_by 'log_index' ASCENDING as a secondary sort key
        messages_query = messages_ref.order_by('timestamp', direction=firestore.Query.ASCENDING) \
                                     .order_by('log_index', direction=firestore.Query.ASCENDING) \
                                     .limit(150) # Increased limit slightly, consider pagination for very long chats

        messages_docs = messages_query.stream()

        history = []
        for msg_doc in messages_docs:
            data = msg_doc.to_dict()
            # Ensure 'parts' is always a list, even if it's stored as a single string or missing
            parts_content = data.get('parts', [])
            if isinstance(parts_content, str): # If 'parts' was accidentally stored as a string
                parts_content = [parts_content]
            elif not isinstance(parts_content, list): # If it's something else or None
                parts_content = [str(parts_content)] if parts_content is not None else []
            
            # Ensure role exists; default to 'model' if missing (though it shouldn't be)
            role = data.get('role', 'model') 
            history.append({'role': role, 'parts': parts_content})

        logging.info(f"Fetched {len(history)} history messages for chat {chat_id}, user {user_id}")
        return jsonify({"history": history})
    except Exception as e:
        # This is where Firebase might return an error if the composite index is missing
        logging.error(f"🔴 Firestore fetch error for chat {chat_id}, user {user_id} (check for missing Firestore index): {e}", exc_info=True)
        if "ensure an index" in str(e).lower(): # Crude check for index error
             return jsonify({"error": f"Database query failed. A Firestore index might be missing. Please check server logs for a link to create it. Details: {e}"}), 500
        return jsonify({"error": "Could not fetch chat history"}), 500


@app.route('/api/chats/<string:chat_id>/message', methods=['POST'])
@check_auth
def post_message_to_chat(chat_id: str):
    """Posts a new message to a specific chat session and gets an AI response."""
    if not db:
        return jsonify({"error": "Database service temporarily unavailable"}), 503

    user_id = request.user_id
    data = request.json
    user_prompt_text = data.get('prompt')
    gemini_history_from_client = data.get('history', []) # This is the history from the client's perspective

    if not user_prompt_text or not user_prompt_text.strip():
        logging.warning(f"User {user_id} sent an empty prompt to chat {chat_id}.")
        return jsonify({"error": "Prompt cannot be empty"}), 400

    logging.info(f"Received prompt for chat {chat_id} from user {user_id}: '{user_prompt_text[:50]}...'")

    # Get AI response using the client-provided history
    response_text, updated_gemini_sdk_history = get_finance_response_backend(user_prompt_text, gemini_history_from_client)

    try:
        chat_session_ref = db.collection(USER_COLLECTION).document(user_id).collection(CHAT_SESSIONS_SUBCOLLECTION).document(chat_id)
        messages_col_ref = chat_session_ref.collection(MESSAGES_SUBCOLLECTION)
        batch = db.batch() # Use a batch for atomic writes of user message, model message, and session update.

        # --- User Message Saving ---
        user_parts_to_save = [user_prompt_text] # The user's current input
        user_doc_ref = messages_col_ref.document() # Auto-generate ID for the new message
        batch.set(user_doc_ref, {
            'role': 'user',
            'parts': user_parts_to_save,
            'timestamp': firestore.SERVER_TIMESTAMP,
            'log_index': 0  # User message comes first in a turn
        })

        # --- Model Message Saving ---
        # Ensure 'response_text' from Gemini is correctly extracted for saving.
        # 'updated_gemini_sdk_history' contains the full conversation history including the latest model response.
        # The last message in 'updated_gemini_sdk_history' should be the model's response.
        model_parts_to_save = [response_text] # Default to the direct response_text
        if updated_gemini_sdk_history and updated_gemini_sdk_history[-1].role == 'model':
             # Extract parts from the SDK's last message object if available and structured correctly
             model_parts_to_save = [p.text for p in updated_gemini_sdk_history[-1].parts if hasattr(p, 'text') and p.text is not None]
             if not model_parts_to_save and response_text: # Fallback if parts extraction fails but response_text is good
                model_parts_to_save = [response_text]

        model_doc_ref = messages_col_ref.document() # Auto-generate ID for the new message
        batch.set(model_doc_ref, {
            'role': 'model',
            'parts': model_parts_to_save,
            'timestamp': firestore.SERVER_TIMESTAMP,
            'log_index': 1  # Model message comes after user message for the same timestamp event
        })
        
        # --- Update Session Metadata (Title and lastUpdatedAt) ---
        session_update_data = {"lastUpdatedAt": firestore.SERVER_TIMESTAMP}
        
        session_doc_snapshot = chat_session_ref.get() # Get current session data
        new_title_to_return = None

        if session_doc_snapshot.exists:
            current_title = session_doc_snapshot.to_dict().get("title", "")
            # Only update title if it's still the default one and this is the first "real" message exchange
            # (i.e., client-side history was empty, implying this is the first prompt for a new chat)
            if not gemini_history_from_client and (current_title.startswith(DEFAULT_CHAT_TITLE_PREFIX) or not current_title.strip()):
                words = user_prompt_text.split()
                generated_title = ' '.join(words[:7]) # Generate title from first few words
                if len(words) > 7:
                    generated_title += "..."
                if generated_title.strip(): # Ensure the generated title is not empty
                    session_update_data["title"] = generated_title
                    new_title_to_return = generated_title # For client update
                    logging.info(f"Updating title for new chat {chat_id} to '{generated_title}' based on first prompt.")
        
        batch.update(chat_session_ref, session_update_data)
        batch.commit() # Commit all batched writes
        logging.info(f"Saved chat turn (user & model) and updated session metadata for chat {chat_id}, user {user_id}.")

        response_payload = {"response": response_text}
        if new_title_to_return:
            # Include updated session info if title changed, so client can update UI
            response_payload["updatedSession"] = {
                "id": chat_id,
                "title": new_title_to_return,
                "lastUpdatedAt": datetime.now(timezone.utc).isoformat() # Approximate client-side update
            }
        return jsonify(response_payload)

    except Exception as e:
        logging.error(f"🔴 Firestore save error for chat {chat_id}, user {user_id}: {e}", exc_info=True)
        # Even if saving fails, the user got the response.
        # Consider if you want to notify the user about the save failure.
        return jsonify({"response": response_text, "error_saving": "Message sent, but failed to save to history."})


@app.route('/api/chats/<string:chat_id>', methods=['DELETE'])
@check_auth
def delete_chat_session(chat_id: str):
    """Deletes a specific chat session and all its messages."""
    if not db:
        return jsonify({"error": "Database service temporarily unavailable"}), 503
    
    user_id = request.user_id
    logging.info(f"Attempting to delete chat session {chat_id} for user {user_id}")

    try:
        session_ref = db.collection(USER_COLLECTION).document(user_id).collection(CHAT_SESSIONS_SUBCOLLECTION).document(chat_id)
        
        session_doc = session_ref.get()
        if not session_doc.exists:
            logging.warning(f"Chat session {chat_id} not found for deletion for user {user_id}.")
            return jsonify({"error": "Chat session not found"}), 404

        # Delete messages in batches (Firestore recommends batching for large deletes)
        messages_ref = session_ref.collection(MESSAGES_SUBCOLLECTION)
        deleted_total_messages = 0
        while True:
            # Fetch a batch of messages to delete. limit() is max 500 for non-transactional batches.
            docs_to_delete_snapshot = messages_ref.limit(500).stream()
            
            batch_delete = db.batch()
            count_in_this_batch = 0
            for doc_snapshot in docs_to_delete_snapshot:
                batch_delete.delete(doc_snapshot.reference)
                count_in_this_batch += 1
            
            if count_in_this_batch == 0:
                break # No more messages to delete in this subcollection
            
            batch_delete.commit() 
            deleted_total_messages += count_in_this_batch
            logging.info(f"Deleted {count_in_this_batch} messages (total {deleted_total_messages} so far) from chat session {chat_id}.")
        
        # After all messages are deleted, delete the session document itself.
        session_ref.delete()
        
        logging.info(f"Successfully deleted chat session {chat_id} and all its {deleted_total_messages} messages for user {user_id}.")
        return jsonify({"message": "Chat session deleted successfully"}), 200

    except Exception as e:
        logging.error(f"🔴 Error deleting chat session {chat_id} for user {user_id}: {e}", exc_info=True)
        return jsonify({"error": "Could not delete chat session"}), 500

@app.route('/api/chats/<string:chat_id>/rename', methods=['PUT'])
@check_auth
def rename_chat_session(chat_id: str):
    """Renames a specific chat session."""
    if not db:
        logging.error("🔴 Database service temporarily unavailable during rename attempt.")
        return jsonify({"error": "Database service temporarily unavailable"}), 503

    user_id = request.user_id
    data = request.json
    new_title = data.get('title')

    if not new_title or not new_title.strip():
        logging.warning(f"User {user_id} attempted to rename chat {chat_id} with an empty title.")
        return jsonify({"error": "New title cannot be empty"}), 400

    trimmed_new_title = new_title.strip()
    logging.info(f"User {user_id} attempting to rename chat {chat_id} to '{trimmed_new_title}'")

    try:
        session_ref = db.collection(USER_COLLECTION).document(user_id).collection(CHAT_SESSIONS_SUBCOLLECTION).document(chat_id)
        
        session_doc = session_ref.get()
        if not session_doc.exists:
            logging.warning(f"Chat session {chat_id} not found for rename attempt by user {user_id}.")
            return jsonify({"error": "Chat session not found"}), 404

        session_ref.update({
            "title": trimmed_new_title,
            "lastUpdatedAt": firestore.SERVER_TIMESTAMP # Update timestamp on rename
        })
        
        logging.info(f"Successfully renamed chat session {chat_id} to '{trimmed_new_title}' for user {user_id}.")
        
        # Fetch the updated document to return current server-resolved timestamp and new title
        updated_doc_snapshot = session_ref.get() 
        updated_data = updated_doc_snapshot.to_dict()
        updated_data['id'] = updated_doc_snapshot.id # Add document ID
        
        # Convert Firestore Timestamps to ISO format strings for JSON response
        if 'createdAt' in updated_data and isinstance(updated_data['createdAt'], datetime):
            updated_data['createdAt'] = updated_data['createdAt'].isoformat()
        if 'lastUpdatedAt' in updated_data and isinstance(updated_data['lastUpdatedAt'], datetime):
                updated_data['lastUpdatedAt'] = updated_data['lastUpdatedAt'].isoformat()
        else: # If lastUpdatedAt was just set by SERVER_TIMESTAMP, it's not a datetime object yet client-side
             updated_data['lastUpdatedAt'] = datetime.now(timezone.utc).isoformat() # approximate


        return jsonify({"message": "Chat session renamed successfully", "session": updated_data}), 200

    except Exception as e:
        logging.error(f"🔴 Error renaming chat session {chat_id} for user {user_id}: {e}", exc_info=True)
        return jsonify({"error": "Could not rename chat session"}), 500


if __name__ == '__main__':
    app_port = int(os.getenv("PORT", 5001)) # Default to 5001 if PORT env var is not set
    # FLASK_DEBUG should be 'true' or 'false' (string) in .env
    is_debug_mode = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    app.run(debug=is_debug_mode, port=app_port, host="0.0.0.0")