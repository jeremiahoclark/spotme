import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import random
import plotly.express as px
from openai import OpenAI

# Show title and description
st.title("💬 Creditworthiness Assessment Chatbot")
st.write(
    "This app assesses creditworthiness based on financial data and uses OpenAI's GPT-3.5 model for analysis. "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys)."
)

# Ask user for their OpenAI API key
openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
    st.stop()

# Create an OpenAI client
client = OpenAI(api_key=openai_api_key)

# Streamlit app
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "Credit Assessment", "Chat"])

    if page == "Dashboard":
        show_financial_summary()
    elif page == "Credit Assessment":
        show_credit_assessment()
    elif page == "Chat":
        show_chat_interface()

def generate_fake_financial_data(start_date, end_date):
    """Generate fake financial data"""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    transactions = []
    categories = [
        "Food and Drink",
        "Shopping",
        "Transportation",
        "Bills and Utilities",
        "Income",
    ]
    
    current_date = start
    while current_date <= end:
        num_transactions = random.randint(1, 5)
        for _ in range(num_transactions):
            category = random.choice(categories)
            amount = round(random.uniform(10, 1000), 2)
            if category == "Income":
                amount = -amount  # Make income negative
            
            transaction = {
                "date": current_date,
                "name": f"Transaction {random.randint(1000, 9999)}",
                "amount": amount,
                "category": category,
            }
            transactions.append(transaction)
        current_date += timedelta(days=1)
    
    return pd.DataFrame(transactions)

def analyze_financial_data(df):
    """Analyze the financial data and return key metrics"""
    total_expenses = df[df['amount'] > 0]['amount'].sum()
    total_income = abs(df[df['amount'] < 0]['amount'].sum())
    net_savings = total_income - total_expenses
    
    category_breakdown = df[df['amount'] > 0].groupby('category')['amount'].sum().sort_values(ascending=False)
    
    return {
        'total_expenses': total_expenses,
        'total_income': total_income,
        'net_savings': net_savings,
        'category_breakdown': category_breakdown
    }

def show_financial_summary():
    st.title("Financial Dashboard")
    
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    df = generate_fake_financial_data(start_date, end_date)
    analysis = analyze_financial_data(df)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Income", f"${analysis['total_income']:.2f}")
    col2.metric("Total Expenses", f"${analysis['total_expenses']:.2f}")
    col3.metric("Net Savings", f"${analysis['net_savings']:.2f}")
    
    st.subheader("Expense Breakdown")
    fig = px.pie(values=analysis['category_breakdown'].values, 
                 names=analysis['category_breakdown'].index,
                 title="Expenses by Category")
    st.plotly_chart(fig)
    
    st.subheader("Transaction History")
    st.dataframe(df[['date', 'name', 'amount', 'category']])

def calculate_credit_score(total_income, total_expenses, net_savings, transaction_history):
    income_to_expense_ratio = total_income / total_expenses if total_expenses > 0 else float('inf')
    savings_rate = net_savings / total_income if total_income > 0 else 0
    
    score = 0
    
    if income_to_expense_ratio > 2:
        score += 40
    elif income_to_expense_ratio > 1.5:
        score += 30
    elif income_to_expense_ratio > 1:
        score += 20
    else:
        score += 10
    
    if savings_rate > 0.2:
        score += 40
    elif savings_rate > 0.1:
        score += 30
    elif savings_rate > 0:
        score += 20
    else:
        score += 10
    
    if len(transaction_history) >= 30:
        score += 20
    elif len(transaction_history) >= 15:
        score += 10
    
    return score

def estimate_credit_limit(credit_score, total_income):
    if credit_score >= 90:
        return min(total_income * 0.5, 50000)
    elif credit_score >= 70:
        return min(total_income * 0.3, 30000)
    elif credit_score >= 50:
        return min(total_income * 0.2, 20000)
    else:
        return min(total_income * 0.1, 10000)

def get_gpt_credit_assessment(financial_data, credit_score, estimated_credit_limit):
    prompt = f"""
    As a financial advisor, analyze the following financial data and provide a credit assessment:

    Total Income: ${financial_data['total_income']:.2f}
    Total Expenses: ${financial_data['total_expenses']:.2f}
    Net Savings: ${financial_data['net_savings']:.2f}
    Credit Score: {credit_score}
    Estimated Credit Limit: ${estimated_credit_limit:.2f}

    Expense Breakdown:
    {', '.join([f"{k}: ${v:.2f}" for k, v in financial_data['category_breakdown'].items()])}

    Please provide:
    1. An overall assessment of the individual's creditworthiness.
    2. Key factors influencing the credit assessment.
    3. Recommendations for improving creditworthiness.
    4. Any potential red flags or areas of concern.

    Format your response in markdown, using appropriate headers and bullet points.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a financial advisor providing credit assessments."},
                {"role": "user", "content": prompt}
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error in generating credit assessment: {str(e)}"

def perform_credit_assessment(df):
    analysis = analyze_financial_data(df)
    
    credit_score = calculate_credit_score(
        analysis['total_income'],
        analysis['total_expenses'],
        analysis['net_savings'],
        df
    )
    
    estimated_credit_limit = estimate_credit_limit(credit_score, analysis['total_income'])
    
    gpt_assessment = get_gpt_credit_assessment(analysis, credit_score, estimated_credit_limit)
    
    return {
        'analysis': analysis,
        'credit_score': credit_score,
        'estimated_credit_limit': estimated_credit_limit,
        'gpt_assessment': gpt_assessment
    }

def show_credit_assessment():
    st.title("Credit Assessment")
    
    start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    df = generate_fake_financial_data(start_date, end_date)
    
    assessment = perform_credit_assessment(df)
    
    col1, col2 = st.columns(2)
    col1.metric("Credit Score", f"{assessment['credit_score']}/100")
    col2.metric("Estimated Credit Limit", f"${assessment['estimated_credit_limit']:.2f}")
    
    st.subheader("GPT Credit Assessment")
    st.markdown(assessment['gpt_assessment'])
    
    st.subheader("Financial Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Income", f"${assessment['analysis']['total_income']:.2f}")
    col2.metric("Total Expenses", f"${assessment['analysis']['total_expenses']:.2f}")
    col3.metric("Net Savings", f"${assessment['analysis']['net_savings']:.2f}")
    
    st.subheader("Expense Breakdown")
    fig = px.pie(values=assessment['analysis']['category_breakdown'].values, 
                 names=assessment['analysis']['category_breakdown'].index,
                 title="Expenses by Category")
    st.plotly_chart(fig)

def show_chat_interface():
    st.title("Chat with Financial Advisor")

    # Create a session state variable to store the chat messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display existing chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about your financial situation"):
        # Store and display the current prompt
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate a response using the OpenAI API
        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )

        # Stream the response to the chat
        with st.chat_message("assistant"):
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()