# Purchase conversation manual regression test

Run the app in deterministic mode (or select the deterministic provider in the UI), start a new chat session, and send the following initial message:

> I want to buy Sony headphones for ₹18,000. I already have a pair of headphones that still works, but the battery has become a little weak. I work on my laptop and attend meetings every day, and I also commute regularly. My friends recently bought premium headphones, so I feel like upgrading too. I am not sure whether I genuinely need a new pair or just want one. Ask me one relevant question at a time, remember my answers, and update the regret-risk score, purchase-pressure score, and buy/wait/don’t-buy recommendation after every answer. Explain what changes in the scores and why. Don't assume I can or cannot afford them because I haven't shared my financial details. At the end, give me a clear recommendation based on the evidence.

Use these suggested answers as the conversation progresses. The assistant should choose relevant questions naturally rather than follow a fixed order:

1. **Usage:** “I use my current headphones for work meetings and commuting almost every day.”
2. **Existing product condition:** “The battery lasts around two hours now, but I can still use them while charging.”
3. **Motivation:** “My main reason is better noise cancellation and comfort. Seeing my friends' new headphones also makes me want them.”
4. **Urgency and alternatives:** “I can wait a month. I haven't checked whether my current headphones can be repaired.”
5. **Financial uncertainty and final recommendation:** “I don't know my budget yet. Please give me your recommendation using only what I've told you.”

After every user answer, record:

- the assistant's exact response;
- regret-risk score and its factor explanation;
- purchase-pressure score and its factor explanation;
- the current buy/wait/don't-buy decision signal;
- facts retained, missing inputs, alternatives, and any changed evidence.

Expected evidence to recognize: daily meetings and commuting are a legitimate use case; the current headphones still function but their battery is a convenience limitation; noise cancellation and comfort are practical potential benefits; social comparison is one motivation but does not prove irrationality; waiting and repair are meaningful alternatives; and affordability remains unknown without a budget or other financial details. The final verdict must follow the application's deterministic scoring rules and evidence, not this test's wording.

Verify that each turn shows one relevant question at most, does not repeatedly ask for facts already supplied, and updates the existing dashboard analysis without losing earlier context.
