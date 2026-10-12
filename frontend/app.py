import requests
import streamlit as st
import os

# 1. PAGE CONFIGURATION


st.set_page_config(
    page_title="Customer Experience Intelligence Platform",
    page_icon="📊",
    layout="wide",
)


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
# 2. DASHBOARD HEADER


st.title("AI-Powered Customer Experience Intelligence Platform")
st.write(
    "Analyze customer sentiment, predict churn risk using deep learning "
    "and discover customer segments using machine learning."
)

# Check backend connection
try:
    health_response = requests.get(
        f"{API_URL}/health",
        timeout=5,
    )
    api_available = health_response.ok
except requests.RequestException:
    api_available = False

st.divider()


# 3. DASHBOARD TABS


sentiment_tab, churn_tab, segment_tab = st.tabs(
    [
        "💬 Sentiment Analysis",
        "⚠️ Churn Prediction",
        "👥 Customer Segmentation",
    ]
)


# 4. SENTIMENT ANALYSIS


with sentiment_tab:
    st.subheader("Customer Review Sentiment")
    st.write("Enter a review to predict its sentiment.")

    review_text = st.text_area(
        "Customer review",
        placeholder="Example: I love this product. Great quality!",
        height=130,
        key="review_text",
    )

    if st.button("Analyze Sentiment", key="sentiment_button"):
        if not review_text.strip():
            st.warning("Please enter a customer review.")
        elif not api_available:
            st.error("Start the FastAPI backend first.")
        else:
            try:
                response = requests.post(
                    f"{API_URL}/predict/sentiment",
                    json={"review_text": review_text},
                    timeout=60,
                )

                if response.ok:
                    result = response.json()

                    st.success(
                        f"Predicted sentiment: "
                        f"{result['predicted_sentiment']}"
                    )

                    st.metric(
                        "Model confidence",
                        f"{result['confidence']:.2%}",
                    )

                    st.write("Class probabilities")
                    st.bar_chart(
                        result["class_probabilities"]
                    )
                else:
                    st.error(
                        f"API error ({response.status_code}): "
                        f"{response.text}"
                    )

            except requests.RequestException as error:
                st.error(f"Could not contact the API: {error}")


# 5. CHURN PREDICTION


with churn_tab:
    st.subheader("Customer Churn Risk")
    st.write("Enter customer details to estimate churn risk.")

    with st.form("churn_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input(
                "Age", min_value=0, max_value=120, value=28, step=1
            )
            location = st.text_input(
                "Location", value="Kerala"
            )
            total_orders = st.number_input(
                "Total orders", min_value=0, value=5, step=1
            )
            avg_order_value = st.number_input(
                "Average order value", min_value=0.0, value=750.0
            )

        with col2:
            days_since_last_order = st.number_input(
                "Days since last order",
                min_value=0,
                value=45, step=1,
            )
            support_tickets = st.number_input(
                "Support tickets", min_value=0, value=2, step=1
            )
            customer_satisfaction = st.slider(
                "Customer satisfaction",
                min_value=1.0,
                max_value=5.0,
                value=4.0, step=0.1,
            )

        churn_submitted = st.form_submit_button(
            "Predict Churn"
        )

    if churn_submitted:
        if not location.strip():
            st.warning("Please enter a location.")
        elif not api_available:
            st.error("Start the FastAPI backend first.")
        else:
            payload = {
                "age": age,
                "location": location.strip(),
                "total_orders": total_orders,
                "avg_order_value": avg_order_value,
                "days_since_last_order": days_since_last_order,
                "support_tickets": support_tickets,
                "customer_satisfaction": customer_satisfaction,
            }

            try:
                response = requests.post(
                    f"{API_URL}/predict/churn",
                    json=payload,
                    timeout=30,
                )

                if response.ok:
                    result = response.json()

                    if result["predicted_churn"] == 1:
                        st.warning(
                            "The model predicts that this customer "
                            "may churn."
                        )
                    else:
                        st.success(
                            "The model does not predict churn "
                            "for this customer."
                        )

                    if "churn_probability" in result:
                        st.metric(
                            "Estimated churn probability",
                            f"{result['churn_probability']:.2%}",
                        )

                    st.caption(
                        "This is a model prediction, not a guarantee "
                        "of future customer behaviour."
                    )
                else:
                    st.error(
                        f"API error ({response.status_code}): "
                        f"{response.text}"
                    )

            except requests.RequestException as error:
                st.error(f"Could not contact the API: {error}")


# 6. CUSTOMER SEGMENTATION


with segment_tab:
    st.subheader("Customer Behaviour Segmentation")
    st.write("Enter customer behaviour data to identify its cluster.")

    with st.form("segmentation_form"):
        total_orders_seg = st.number_input(
            "Total orders",
            min_value=0,
            value=5, step=1,
            key="seg_orders",
        )
        avg_order_value_seg = st.number_input(
            "Average order value",
            min_value=0.0,
            value=750.0, step=1.0,
            key="seg_value",
        )
        days_since_last_order_seg = st.number_input(
            "Days since last order",
            min_value=0,
            value=45, step=1,
            key="seg_days",
        )
        support_tickets_seg = st.number_input(
            "Support tickets",
            min_value=0,
            value=2, step=1,
            key="seg_tickets",
        )
        customer_satisfaction_seg = st.slider(
            "Customer satisfaction",
            min_value=1.0,
            max_value=5.0,
            value=4.0, step=0.1,
            key="seg_satisfaction",
        )

        segment_submitted = st.form_submit_button(
            "Identify Customer Segment"
        )

    if segment_submitted:
        if not api_available:
            st.error("Start the FastAPI backend first.")
        else:
            payload = {
                "total_orders": total_orders_seg,
                "avg_order_value": avg_order_value_seg,
                "days_since_last_order": days_since_last_order_seg,
                "support_tickets": support_tickets_seg,
                "customer_satisfaction": customer_satisfaction_seg,
            }

            try:
                response = requests.post(
                    f"{API_URL}/predict/segment",
                    json=payload,
                    timeout=30,
                )

                if response.ok:
                    result = response.json()

                    st.success(
                        f"Customer assigned to Cluster "
                        f"{result['customer_cluster']}"
                    )

                    st.info(result["message"])

                    st.caption(
                        "Cluster numbers identify groups learned by "
                        "KMeans. They do not inherently indicate a "
                        "good or bad customer."
                    )
                else:
                    st.error(
                        f"API error ({response.status_code}): "
                        f"{response.text}"
                    )

            except requests.RequestException as error:
                st.error(f"Could not contact the API: {error}")

