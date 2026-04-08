"""
Test question bank for each chatbot domain (89 questions across 9 sections).

Run a section:       python test_sections.py "Calendar"
Generate gold data:  python generate_gold_answers.py
Full 6-metric eval:  python evaluate_section_metrics.py --k 3
"""

import json
import logging
from ui.app_with_clarification_memory import get_answer

logging.getLogger().setLevel(logging.WARNING)

questions = {
    "Calendar": [
        "When does Spring 2026 semester start?",
        "When are Spring 2026 final exams?",
        "When is Spring 2026 commencement?",
        "When does Summer 1 2026 start?",
        "When is the last day to withdraw from a Spring 2026 course?",
        "When are Spring 2026 midterm grades due?",
        "How long is the Spring 2026 final exam period?",
        "How long is the Spring 2026 semester?",
        "If I apply for graduation in Summer 2026 but want to walk in the Spring 2026 ceremony, when do I apply and which ceremony?",
        "If I withdraw from a Spring course on April 10, is that before or after the withdrawal deadline?",
        "If the add/drop deadline is January 20 and I register January 21, what happens?",
        "I have a friend visiting from out of town next week. Are there any days in April when IIT has no classes?",
        "My lease ends in May. When exactly do spring semester classes finish?",
        "I have a wedding to attend in late November. Will I miss any class days?"
    ],
    "Contact": [
        "Where is the Registrar's office located?",
        "What is the fax number for the Department of Physics?",
        "How do I contact the Pritzker Institute of Biomedical Science?",
        "Who should I contact if my transcript request has an issue?",
        "Who should I contact about registration problems?",
        "Who should I contact if I have a hold on my account?",
        "I want to contact someone at the Wanger Institute. What is their number?",
        "I want to take a semester off. What is the process?",
        "If my advisor approval is missing during registration, who should I contact?"
    ],
    "Policy": [
        "How many credits do graduate students need to be full-time?",
        "How many credits do undergraduate students need to be full-time?",
        "What is the pass/fail policy at IIT?",
        "I am a graduate student taking 8 credits this semester — am I full-time?",
        "What is the difference between withdrawing and dropping a course?",
        "I failed a class and want to retake it. Will both grades show on my transcript?",
        "I'm graduating but haven't registered for any courses this semester. Is that a problem?",
        "How many courses can I take pass/fail as an undergraduate?",
        "What is the max pass/fail courses I can take as an undergrad, and if I take a required major course pass/fail, will it count toward my degree?",
        "As a coterminal student, am I considered undergraduate or graduate for full-time status purposes, and how does tuition work?",
        "I'm a grad student who got a C in a required course — can I repeat it, will the old grade be replaced, and how does it affect my GPA?",
        "Can I work on a research project with a professor and keep the intellectual property?"
    ],
    "Registration": [
        "How do I request a late registration?",
        "How do I add a course during the add/drop period?",
        "When does registration open for Fall 2026?",
        "Who do I contact if I have a registration issue?",
        "What happens if I try to register after the add/drop deadline?",
        "I have a hold on my account and registration opens tomorrow — what types of holds exist and how do I find out what mine is?",
        "If registration opens tomorrow but I have a financial hold, can I still register?",
        "If I miss the add/drop deadline, can I still add a class?"
    ],
    "Transcripts": [
        "How do I get an official transcript from IIT?",
        "Can I order an official transcript online?",
        "Is there a fee for ordering an official transcript?",
        "How long does it take to process a transcript request?",
        "Can alumni request transcripts through the same system?",
        "Can transcripts be sent electronically?",
        "If I graduated last year and need a transcript for a job application, how do I request it?",
        "If my transcript needs to be sent to multiple schools, how do I request that?",
        "If I need a transcript urgently, is there an expedited option?"
    ],
    "Tuition": [
        "What is the tuition per credit hour for graduate students on the Mies campus?",
        "What is the tuition per credit hour for undergraduate students?",
        "Where can I find current tuition rates?",
        "What happens to my tuition if I drop a class after the add/drop deadline?",
        "Do I get a refund if I drop a course before the add/drop deadline?",
        "How is tuition calculated for part-time students?",
        "If I drop from 12 credits to 9 credits mid-semester, how does that affect my tuition?",
        "If I withdraw from all classes after the fourth week, do I receive a tuition refund?",
        "If I drop a course but remain full-time, does my tuition change?",
        "I'm starting the MDes program. What will I pay per semester?",
        "Are there any fees specific to Chicago-Kent students beyond tuition?"
    ],
    "Financial Aid": [
        "Do undergraduate students need to be full-time to receive financial aid?",
        "Where can I find financial aid policies?",
        "What happens to my financial aid if I drop below full-time status?",
        "Does withdrawing from a course affect financial aid eligibility?",
        "Can dropping a course require repayment of financial aid?",
        "I receive financial aid and I'm thinking of dropping below full-time status mid-semester — what happens to my aid and is there any tuition refund?",
        "If I withdraw from all courses mid-semester, how does it affect my financial aid?",
        "If I drop from full-time to part-time, will my grants or loans change?"
    ],
    "Transfer Credit": [
        "Can graduate students transfer credits into a master's program?",
        "Do transfer credits count toward degree requirements?",
        "Who approves transfer credits?",
        "What is the maximum number of transfer credits allowed for a master's program?",
        "Do transfer credits affect GPA calculations?",
        "When should transfer credits be submitted?",
        "I transferred 6 credits into a master's program — how many more transfer credits can I bring in, and could this affect my graduation date?",
        "If I transfer the maximum allowed credits, could I graduate earlier?",
        "Can transfer credits from another graduate program be applied to my degree?"
    ],
    "Multi-topic": [
        "If I take 9 credits as a graduate student, am I considered full-time?",
        "If I take 12 credits as an undergraduate student, am I considered full-time?",
        "What happens if I drop a course before the add/drop deadline?",
        "I'm a graduate student taking 8 credits — am I full-time and will it affect my tuition?",
        "If I drop a course after the add/drop deadline, will it appear on my transcript?",
        "If I withdraw from a course, does it affect my GPA?",
        "I'm an F-1 grad student taking 9 credits in Spring 2026 — am I full-time, and what happens to my visa status if I drop one 3-credit course after the add/drop deadline?",
        "I'm an undergraduate student with 12 credits — if I drop a 3-credit course after the add/drop deadline, am I still full-time and will I get a tuition refund?",
        "If I withdraw from all my classes in Spring 2026 after the 4th week, do I get any tuition refund and what grade appears on my transcript?"
    ]
}

def run_section(section_name):
    if section_name not in questions:
        print(f"Section {section_name} not found.")
        return
        
    print(f"\n==============================================")
    print(f"RUNNING SECTION: {section_name.upper()}")
    print(f"==============================================\n")
    
    for count, q in enumerate(questions[section_name], 1):
        print(f"Q{count}: {q}")
        
        # Reset chat history for each question to isolate testing
        reply, sources, route_details, is_clarification, clarification_msg, clarifying_domain, _clar_opts = get_answer(
            query=q,
            chat_history=[]
        )
        
        if is_clarification:
            print(f"---> [CLARIFICATION NEEDED]")
            print(f"---> Domain: {clarifying_domain}")
            print(f"---> Message: {clarification_msg}")
        else:
            print(f"---> [ANSWER]")
            # Truncate answer for readability in CLI
            formatted_reply = reply.replace("\n", " ")
            print(f"---> {formatted_reply[:250]}..." if len(formatted_reply) > 250 else f"---> {formatted_reply}")
            
            if sources:
                print(f"---> Sources: {len(sources)}")
        
        if route_details and "domains" in route_details:
            print(f"---> Routed to: {', '.join(route_details['domains'])}")
            
        print("-" * 50)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run_section(sys.argv[1])
    else:
        print(f"Available sections: {', '.join(questions.keys())}")
        print("Usage: python test_sections.py \"Section Name\"")
