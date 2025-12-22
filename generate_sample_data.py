"""
Generate sample dataset for testing and demonstration
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import config


# Sample ticket templates for each category
TICKET_TEMPLATES = {
    'Bug Report': [
        ("Application crashes when opening reports", 
         "Every time I try to open the monthly report, the application crashes. This started happening after the last update."),
        ("Login button not responding", 
         "The login button on the homepage doesn't respond when clicked. I've tried multiple browsers."),
        ("Data not saving correctly", 
         "When I fill out the form and click save, the data disappears. This is very frustrating."),
        ("Error message appearing repeatedly", 
         "I keep getting an error message 'Connection timeout' even though my internet is working fine."),
        ("Feature broken after update", 
         "The export feature stopped working after the recent update. It was working fine before."),
    ],
    'Feature Request': [
        ("Add dark mode option", 
         "Please add a dark mode theme option. It would be easier on the eyes during night work."),
        ("Request for mobile app", 
         "It would be great to have a mobile app version. Many users work on the go."),
        ("Export to Excel feature", 
         "Could you add an option to export reports directly to Excel format? Currently only PDF is available."),
        ("Bulk actions request", 
         "It would be helpful to have bulk selection and actions for managing multiple items at once."),
        ("Customizable dashboard", 
         "Please allow users to customize their dashboard layout and widgets."),
    ],
    'Technical Issue': [
        ("Cannot connect to server", 
         "I'm unable to connect to the server. Getting connection refused errors. Firewall settings are correct."),
        ("API integration problem", 
         "The API integration is not working properly. Getting 500 errors when making requests."),
        ("Database connection timeout", 
         "Experiencing frequent database connection timeouts. This is affecting our operations."),
        ("Performance degradation", 
         "The system has become very slow recently. Response times have increased significantly."),
        ("SSL certificate issue", 
         "There's an SSL certificate error when accessing the secure portal. Certificate seems expired."),
    ],
    'Billing Inquiry': [
        ("Question about invoice charges", 
         "I received an invoice but I don't understand some of the charges. Can someone explain?"),
        ("Payment method update", 
         "I need to update my payment method. My credit card expired and I want to use a new one."),
        ("Refund request", 
         "I canceled my subscription last month but haven't received a refund yet. When can I expect it?"),
        ("Billing cycle question", 
         "When does my billing cycle reset? I want to know when the next charge will occur."),
        ("Invoice download issue", 
         "I'm unable to download my invoice. The download link doesn't work."),
    ],
    'Account Management': [
        ("Password reset request", 
         "I forgot my password and need to reset it. The reset link in email is not working."),
        ("Account access issue", 
         "I can't access my account. It says my account is locked. I haven't done anything wrong."),
        ("Update profile information", 
         "I need to update my profile information but the save button is not working."),
        ("Account deletion request", 
         "I want to delete my account permanently. Please guide me through the process."),
        ("Permission access problem", 
         "I should have admin access but I'm only seeing user-level permissions. Please fix this."),
    ]
}

PRIORITIES = ['Low', 'Medium', 'High', 'Critical']


def generate_sample_tickets(n_tickets=500):
    """Generate sample support tickets"""
    tickets = []
    
    # Ensure balanced distribution across categories
    tickets_per_category = n_tickets // len(config.CATEGORIES)
    remaining = n_tickets % len(config.CATEGORIES)
    
    ticket_id = 1
    base_date = datetime.now() - timedelta(days=90)
    
    for category in config.CATEGORIES:
        num_tickets = tickets_per_category + (1 if remaining > 0 else 0)
        remaining -= 1
        
        templates = TICKET_TEMPLATES[category]
        
        for i in range(num_tickets):
            # Select random template
            subject, description = random.choice(templates)
            
            # Add some variation to make tickets more realistic
            if random.random() < 0.3:
                description += " " + random.choice([
                    "Please help as soon as possible.",
                    "This is urgent.",
                    "Thank you for your assistance.",
                    "I've tried troubleshooting but nothing works.",
                    "This is affecting my work.",
                ])
            
            # Generate random timestamp
            days_ago = random.randint(0, 90)
            hours_ago = random.randint(0, 23)
            timestamp = base_date + timedelta(days=days_ago, hours=hours_ago)
            
            # Assign priority (some categories more likely to be high priority)
            if category == 'Bug Report':
                priority_weights = [0.1, 0.2, 0.4, 0.3]  # More high/critical
            elif category == 'Technical Issue':
                priority_weights = [0.05, 0.15, 0.5, 0.3]
            elif category == 'Billing Inquiry':
                priority_weights = [0.3, 0.4, 0.2, 0.1]
            else:
                priority_weights = [0.2, 0.3, 0.3, 0.2]
            
            priority = random.choices(PRIORITIES, weights=priority_weights)[0]
            
            tickets.append({
                'Ticket_ID': ticket_id,
                'Subject': subject,
                'Description': description,
                'Category': category,
                'Priority': priority,
                'Timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S')
            })
            
            ticket_id += 1
    
    # Shuffle tickets
    random.shuffle(tickets)
    
    # Reassign sequential IDs
    for i, ticket in enumerate(tickets, 1):
        ticket['Ticket_ID'] = i
    
    return pd.DataFrame(tickets)


def main():
    """Generate and save sample dataset"""
    print("=" * 60)
    print("Generating Sample Support Ticket Dataset")
    print("=" * 60)
    
    # Generate sample data
    print("\nGenerating 500 sample tickets...")
    df = generate_sample_tickets(n_tickets=500)
    
    # Save to CSV
    output_path = config.SAMPLE_DATASET_PATH
    df.to_csv(output_path, index=False)
    
    print(f"\nSample dataset saved to: {output_path}")
    print(f"Total tickets: {len(df)}")
    print("\nCategory distribution:")
    print(df['Category'].value_counts())
    print("\nPriority distribution:")
    print(df['Priority'].value_counts())
    
    print("\n" + "=" * 60)
    print("Sample dataset generated successfully!")
    print("=" * 60)
    print("\nTo use this dataset for training, update config.py:")
    print(f"  DATASET_PATH = '{output_path}'")
    print("=" * 60)


if __name__ == "__main__":
    main()

