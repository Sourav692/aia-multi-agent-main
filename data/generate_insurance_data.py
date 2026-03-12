"""
AIA Insurance Multi-Agent System — Synthetic Data Generator
Generates realistic insurance data for Claims, Policies, Customers, Products, and Agents.
"""

import csv
import random
import json
from datetime import datetime, timedelta
import os

random.seed(42)

# --- Configuration ---
NUM_CUSTOMERS = 2000
NUM_POLICIES = 3000
NUM_CLAIMS = 5000
NUM_PRODUCTS = 20
NUM_AGENTS = 50
NUM_POLICY_DOCUMENTS = 200  # For RAG / unstructured search

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Reference Data ---
REGIONS = ["Hong Kong", "Singapore", "Thailand", "Malaysia", "Indonesia", "Philippines", "Vietnam", "South Korea", "Taiwan", "Mainland China"]
REGION_WEIGHTS = [0.20, 0.15, 0.12, 0.10, 0.10, 0.08, 0.07, 0.07, 0.06, 0.05]

PRODUCT_CATEGORIES = ["Life", "Health", "Motor", "Property", "Travel", "Investment-Linked", "Critical Illness", "Accident"]
CLAIM_STATUSES = ["Approved", "Pending", "Under Investigation", "Rejected", "Settled"]
CLAIM_TYPES = ["Hospitalization", "Outpatient", "Surgery", "Death Benefit", "Total Permanent Disability", "Critical Illness", "Motor Accident", "Property Damage", "Travel Cancellation", "Maternity"]
POLICY_STATUSES = ["Active", "Lapsed", "Surrendered", "Matured", "Cancelled"]
GENDERS = ["Male", "Female"]
CHANNELS = ["Agency", "Bancassurance", "Digital", "Broker", "Direct"]
CUSTOMER_SEGMENTS = ["Mass", "Mass Affluent", "High Net Worth", "Ultra High Net Worth"]

HOSPITALS = [
    "Queen Mary Hospital", "Mount Elizabeth Hospital", "Bumrungrad International",
    "Gleneagles Hospital", "Prince of Wales Hospital", "Raffles Hospital",
    "Bangkok Hospital", "Pantai Hospital", "Siloam Hospital", "St. Luke's Medical Center"
]

FIRST_NAMES = [
    "Wei", "Min", "Jia", "Hui", "Xin", "Yan", "Li", "Mei", "Ling", "Fang",
    "Ahmad", "Siti", "Raj", "Priya", "Tan", "Lee", "Kim", "Park", "Chen", "Wang",
    "John", "Sarah", "Michael", "Emma", "David", "Lisa", "Robert", "Jennifer",
    "Somchai", "Nguyen", "Maria", "Jose", "Arjun", "Ananya", "Rizal", "Dewi"
]

LAST_NAMES = [
    "Wong", "Lim", "Tan", "Lee", "Chen", "Ng", "Chan", "Ho", "Lin", "Huang",
    "Kumar", "Singh", "Rahman", "Suzuki", "Takahashi", "Park", "Kim", "Nguyen",
    "Pham", "Garcia", "Santos", "Reyes", "Sharma", "Patel", "Wijaya", "Sari"
]


def random_date(start_year, end_year):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))


def generate_products():
    products = []
    product_id = 1
    variants_per_cat = max(2, NUM_PRODUCTS // len(PRODUCT_CATEGORIES))
    for cat in PRODUCT_CATEGORIES:
        for variant in range(1, variants_per_cat + 1):
            if product_id > NUM_PRODUCTS:
                break
            base_premium = {
                "Life": random.uniform(500, 5000),
                "Health": random.uniform(300, 3000),
                "Motor": random.uniform(200, 1500),
                "Property": random.uniform(400, 2000),
                "Travel": random.uniform(50, 500),
                "Investment-Linked": random.uniform(1000, 10000),
                "Critical Illness": random.uniform(400, 4000),
                "Accident": random.uniform(100, 800),
            }[cat]
            tier_name = ["Plus", "Premium", "Elite"][variant % 3]
            products.append({
                "product_id": f"PROD-{product_id:03d}",
                "product_name": f"AIA {cat} {tier_name} {random.choice(['Plan', 'Shield', 'Protect', 'Cover'])}",
                "category": cat,
                "sub_category": f"{cat} - Tier {variant}",
                "base_annual_premium_usd": round(base_premium, 2),
                "max_coverage_usd": round(base_premium * random.uniform(50, 200), 2),
                "min_entry_age": 18 if cat != "Travel" else 0,
                "max_entry_age": 65 if cat in ["Life", "Critical Illness"] else 75,
                "policy_term_years": random.choice([5, 10, 15, 20, 25, 30]) if cat in ["Life", "Investment-Linked"] else 1,
                "launch_date": random_date(2015, 2023).strftime("%Y-%m-%d"),
                "is_active": random.choice([True, True, True, False]),
                "region_availability": ",".join(random.sample(REGIONS, random.randint(3, len(REGIONS)))),
            })
            product_id += 1
            if product_id > NUM_PRODUCTS:
                break
        if product_id > NUM_PRODUCTS:
            break
    return products[:NUM_PRODUCTS]


def generate_agents():
    agents = []
    for i in range(1, NUM_AGENTS + 1):
        region = random.choices(REGIONS, weights=REGION_WEIGHTS, k=1)[0]
        agents.append({
            "agent_id": f"AGT-{i:04d}",
            "agent_name": f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}",
            "region": region,
            "channel": random.choices(CHANNELS, weights=[0.40, 0.25, 0.15, 0.10, 0.10], k=1)[0],
            "years_experience": random.randint(1, 25),
            "certification_level": random.choice(["Associate", "Professional", "Senior", "Executive"]),
            "active_policies_count": random.randint(10, 500),
            "total_premium_sold_usd": round(random.uniform(50000, 5000000), 2),
            "hire_date": random_date(2000, 2023).strftime("%Y-%m-%d"),
            "is_active": random.choice([True, True, True, True, False]),
        })
    return agents


def generate_customers():
    customers = []
    for i in range(1, NUM_CUSTOMERS + 1):
        region = random.choices(REGIONS, weights=REGION_WEIGHTS, k=1)[0]
        age = random.randint(20, 75)
        segment = random.choices(
            CUSTOMER_SEGMENTS,
            weights=[0.50, 0.30, 0.15, 0.05],
            k=1
        )[0]
        annual_income = {
            "Mass": random.uniform(15000, 50000),
            "Mass Affluent": random.uniform(50000, 200000),
            "High Net Worth": random.uniform(200000, 1000000),
            "Ultra High Net Worth": random.uniform(1000000, 10000000),
        }[segment]
        customers.append({
            "customer_id": f"CUST-{i:05d}",
            "first_name": random.choice(FIRST_NAMES),
            "last_name": random.choice(LAST_NAMES),
            "gender": random.choice(GENDERS),
            "date_of_birth": random_date(1950, 2004).strftime("%Y-%m-%d"),
            "age": age,
            "region": region,
            "city": f"{region} City",
            "segment": segment,
            "annual_income_usd": round(annual_income, 2),
            "marital_status": random.choice(["Single", "Married", "Divorced", "Widowed"]),
            "dependents": random.randint(0, 5),
            "occupation": random.choice(["Professional", "Business Owner", "Executive", "Student", "Retired", "Self-Employed", "Government"]),
            "customer_since": random_date(2005, 2024).strftime("%Y-%m-%d"),
            "preferred_channel": random.choice(CHANNELS),
            "nps_score": random.randint(1, 10),
            "total_policies": random.randint(1, 6),
            "lifetime_premium_usd": round(random.uniform(1000, 200000), 2),
            "risk_profile": random.choice(["Conservative", "Moderate", "Aggressive"]),
        })
    return customers


def generate_policies(customers, products, agents):
    policies = []
    for i in range(1, NUM_POLICIES + 1):
        customer = random.choice(customers)
        product = random.choice(products)
        agent = random.choice(agents)
        start_date = random_date(2018, 2024)
        term = product["policy_term_years"]
        end_date = start_date + timedelta(days=365 * term)
        premium_multiplier = random.uniform(0.7, 2.0)
        annual_premium = round(product["base_annual_premium_usd"] * premium_multiplier, 2)
        sum_assured = round(annual_premium * random.uniform(20, 100), 2)

        status = random.choices(
            POLICY_STATUSES,
            weights=[0.65, 0.10, 0.08, 0.07, 0.10],
            k=1
        )[0]

        policies.append({
            "policy_id": f"POL-{i:06d}",
            "customer_id": customer["customer_id"],
            "product_id": product["product_id"],
            "agent_id": agent["agent_id"],
            "policy_status": status,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "annual_premium_usd": annual_premium,
            "sum_assured_usd": sum_assured,
            "payment_frequency": random.choice(["Monthly", "Quarterly", "Semi-Annual", "Annual"]),
            "payment_method": random.choice(["Credit Card", "Bank Transfer", "Auto-Debit", "Cash"]),
            "underwriting_class": random.choice(["Standard", "Preferred", "Substandard", "Declined"]),
            "riders": ",".join(random.sample(["Waiver of Premium", "Accidental Death", "Hospital Cash", "CI Rider", "Disability"], random.randint(0, 3))),
            "channel": agent["channel"],
            "region": customer["region"],
        })
    return policies


def generate_claims(policies, customers, products):
    claims = []
    # Only policies with Active or Matured status can have claims
    eligible_policies = [p for p in policies if p["policy_status"] in ["Active", "Matured"]]

    for i in range(1, NUM_CLAIMS + 1):
        policy = random.choice(eligible_policies)
        customer = next(c for c in customers if c["customer_id"] == policy["customer_id"])
        product = next(p for p in products if p["product_id"] == policy["product_id"])

        claim_date = random_date(2022, 2025)
        claim_type = random.choice(CLAIM_TYPES)

        # Claim amount based on type
        amount_ranges = {
            "Hospitalization": (1000, 50000),
            "Outpatient": (50, 2000),
            "Surgery": (5000, 100000),
            "Death Benefit": (50000, 500000),
            "Total Permanent Disability": (30000, 300000),
            "Critical Illness": (20000, 200000),
            "Motor Accident": (500, 30000),
            "Property Damage": (1000, 50000),
            "Travel Cancellation": (100, 5000),
            "Maternity": (2000, 15000),
        }
        low, high = amount_ranges.get(claim_type, (500, 10000))
        claim_amount = round(random.uniform(low, high), 2)
        approved_amount = round(claim_amount * random.uniform(0.5, 1.0), 2)

        status = random.choices(
            CLAIM_STATUSES,
            weights=[0.45, 0.20, 0.10, 0.10, 0.15],
            k=1
        )[0]

        processing_days = random.randint(1, 90) if status != "Pending" else None

        # Fraud indicators
        is_suspicious = random.random() < 0.05
        fraud_score = round(random.uniform(0.6, 0.99), 3) if is_suspicious else round(random.uniform(0.0, 0.3), 3)

        claims.append({
            "claim_id": f"CLM-{i:06d}",
            "policy_id": policy["policy_id"],
            "customer_id": policy["customer_id"],
            "product_id": policy["product_id"],
            "claim_type": claim_type,
            "claim_date": claim_date.strftime("%Y-%m-%d"),
            "claim_amount_usd": claim_amount,
            "approved_amount_usd": approved_amount if status in ["Approved", "Settled"] else 0,
            "claim_status": status,
            "region": customer["region"],
            "hospital_provider": random.choice(HOSPITALS) if claim_type in ["Hospitalization", "Surgery", "Maternity", "Outpatient"] else None,
            "diagnosis_code": f"ICD-{random.choice('ABCDEFGH')}{random.randint(10, 99)}.{random.randint(0, 9)}",
            "processing_days": processing_days,
            "adjuster_notes": random.choice([
                "Standard claim, all documents verified.",
                "Requires additional medical records.",
                "Pre-authorization confirmed.",
                "Multiple claims from same provider - flagged for review.",
                "Claim within policy coverage limits.",
                "Waiting period not yet elapsed.",
                "Coverage confirmed, processing payment.",
                None
            ]),
            "fraud_score": fraud_score,
            "is_suspicious": is_suspicious,
            "submitted_via": random.choice(["Mobile App", "Web Portal", "Agent", "Email", "Walk-in"]),
            "settlement_date": (claim_date + timedelta(days=processing_days)).strftime("%Y-%m-%d") if processing_days and status in ["Approved", "Settled"] else None,
        })
    return claims


def generate_policy_documents(products):
    """Generate synthetic policy document metadata for RAG / Vector Search."""
    docs = []
    doc_types = ["Policy Wording", "Product Disclosure Sheet", "Benefit Schedule", "Exclusion List", "Claims Procedure Guide", "FAQ", "Underwriting Guidelines"]

    for i in range(1, NUM_POLICY_DOCUMENTS + 1):
        product = random.choice(products)
        doc_type = random.choice(doc_types)

        # Generate a realistic text chunk for embedding
        content_templates = {
            "Policy Wording": f"This {product['product_name']} policy provides coverage for {product['category'].lower()} insurance. "
                             f"The maximum coverage amount is USD {product['max_coverage_usd']:,.2f}. "
                             f"Entry age ranges from {product['min_entry_age']} to {product['max_entry_age']} years. "
                             f"The policy term is {product['policy_term_years']} year(s). "
                             f"Available in the following regions: {product['region_availability']}.",
            "Product Disclosure Sheet": f"{product['product_name']} is a {product['category']} insurance product designed to provide "
                                        f"financial protection. Base annual premium starts from USD {product['base_annual_premium_usd']:,.2f}. "
                                        f"Key benefits include coverage for hospitalization, surgery, and outpatient treatments.",
            "Benefit Schedule": f"The benefit schedule for {product['product_name']} includes: "
                               f"Room & Board - up to USD {random.randint(200, 1000)}/day, "
                               f"ICU - up to USD {random.randint(500, 2000)}/day, "
                               f"Surgical fees - up to USD {random.randint(5000, 50000)}, "
                               f"Annual limit - USD {product['max_coverage_usd']:,.2f}.",
            "Exclusion List": f"The following are excluded from {product['product_name']} coverage: "
                             f"pre-existing conditions within the first 12 months, self-inflicted injuries, "
                             f"participation in hazardous activities, cosmetic surgery, "
                             f"dental treatment (unless caused by accident), and war or terrorism.",
            "Claims Procedure Guide": f"To file a claim under {product['product_name']}: "
                                      f"1. Notify AIA within 30 days of the event. "
                                      f"2. Complete the claim form and attach supporting documents. "
                                      f"3. Submit via AIA mobile app, web portal, or your agent. "
                                      f"4. Claims are typically processed within 5-15 business days. "
                                      f"5. Payment is made via bank transfer to the registered account.",
            "FAQ": f"Frequently asked questions about {product['product_name']}: "
                   f"Q: What is the waiting period? A: {random.choice(['30 days', '60 days', '90 days'])} for illness-related claims. "
                   f"Q: Can I add riders? A: Yes, riders such as Waiver of Premium and Critical Illness are available. "
                   f"Q: What is the grace period for premium payment? A: {random.choice(['30 days', '31 days'])} from the due date.",
            "Underwriting Guidelines": f"Underwriting guidelines for {product['product_name']}: "
                                       f"Standard class: BMI 18.5-27.5, no major pre-existing conditions. "
                                       f"Preferred class: BMI 18.5-25, excellent health history, non-smoker. "
                                       f"Substandard class: May require additional premium loading of {random.randint(25, 100)}%.",
        }

        docs.append({
            "document_id": f"DOC-{i:04d}",
            "product_id": product["product_id"],
            "document_type": doc_type,
            "title": f"{product['product_name']} - {doc_type}",
            "content": content_templates.get(doc_type, f"Document content for {product['product_name']}."),
            "category": product["category"],
            "region": random.choice(product["region_availability"].split(",")),
            "language": random.choice(["English", "Chinese", "Thai", "Malay", "Vietnamese"]),
            "version": f"v{random.randint(1, 5)}.{random.randint(0, 9)}",
            "effective_date": random_date(2020, 2024).strftime("%Y-%m-%d"),
            "last_updated": random_date(2023, 2025).strftime("%Y-%m-%d"),
        })
    return docs


def write_csv(data, filename):
    filepath = os.path.join(OUTPUT_DIR, filename)
    if not data:
        return
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
    print(f"  Written {len(data)} rows to {filename}")


def main():
    print("Generating AIA Insurance synthetic data...")

    products = generate_products()
    write_csv(products, "products.csv")

    agents = generate_agents()
    write_csv(agents, "agents.csv")

    customers = generate_customers()
    write_csv(customers, "customers.csv")

    policies = generate_policies(customers, products, agents)
    write_csv(policies, "policies.csv")

    claims = generate_claims(policies, customers, products)
    write_csv(claims, "claims.csv")

    docs = generate_policy_documents(products)
    write_csv(docs, "policy_documents.csv")

    print(f"\nData generation complete!")
    print(f"  Products: {len(products)}")
    print(f"  Agents: {len(agents)}")
    print(f"  Customers: {len(customers)}")
    print(f"  Policies: {len(policies)}")
    print(f"  Claims: {len(claims)}")
    print(f"  Policy Documents: {len(docs)}")


if __name__ == "__main__":
    main()
