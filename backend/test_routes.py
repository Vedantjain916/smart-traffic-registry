from app import app

with app.test_request_context():
    print("Registered routes:")
    for rule in app.url_map.iter_rules():
        print(f"- {rule.rule} [{', '.join(rule.methods)}]")
