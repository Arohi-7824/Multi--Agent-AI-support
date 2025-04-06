from agents.intent_agent import predict_intent
from agents.response_agent import generate_response


print("🤖 Multi-Agent AI Support System\nType 'exit' to quit.\n")

while True:
    user_msg = input("User: ")
    if user_msg.lower() == "exit":
        break

    try:
        # Step 1: Predict intent
        intent = predict_intent(user_msg)

        # Step 2: Get response from trained dataset
        reply = generate_response(user_msg, intent)

        print(f"AI (Intent: {intent}): {reply}\n")

    except Exception as e:
        print(f"❌ Error: {e}\n")
