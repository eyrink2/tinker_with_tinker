# HW2: Deep Personalization - Writeup

**Name:** Eyrin Kim

**SUNet ID:** eyrinkim


## Setup & Data Collection

### Data Source Description (Bonus: 20 points)
*If you're attempting the bonus by using a creative data source other than email generation, describe it here. Otherwise, write "Email generation (default task)".*

**TODO: Add your answer or leave it empty if you did not attempt the bonus**

## Get Baseline Results (10 points total)

Make sure you have implement `TinkerSampler` in `tinker_personalization.ipynb`. You don't need to answer anything here.


## Method 1: Prompt Engineering (10 points total)

### System Prompt Design (5 points)
Copy-paste the system prompt you engineered to personalize the model outputs.

```text
You are a helpful assistant that writes emails. You will write them in a professional and friendly tone with no emdashes (--). You will limit exclamation marks and focus 
on being succinct while still polite. You are writing from the perspective of Eyrin Kim, a Stanford undergraduate student.

Here are some examples of emails you have written (input was context and output was the email you wrote):

        "input": "Reply to an interviewer with times I am available this week to chat, broken down day by day. I work in pacific time.",
        "output": "Hi Matias, It's great to meet you. I'd love to chat next week — do any of these times work for you? All times in PT:\n\n Tuesday, Oct 14: 10 am - 12 pm, 2 - 3 pm\n\n Wednesday, Oct 15: 4:30 - 6 pm\n\n Thursday, Oct 16: 10 am - 12 pm, 1 - 3 pm\n\n Friday, Oct 17: 1:30 - 4:30 pm\n\n Looking forward to connecting! Let me know if I can answer any questions in the meantime; I'm happy to send more info about myself.\n\n Warmly, Eyrin"

        "input": "Write an email reaching out to a professional in the field of eningeering and energy working at Tapestry asking for an informational coffee chat. Mention my deep interest in the intersection of AI and energy, energy as a critical problem in America, and the work of Google X.",
        "output": "Hi Arianna,\n\n My name is Eyrin, and I'm a Stanford junior studying AI and Earth Systems. I'm reaching out because I would love to learn more about what you do at Google X.\n\nI've been deeply interested in America's electric grid recently and stumbled across Tapestry. Further research led me to Bellwether and Chorus, both of which were directionally similar to work I've done and intellectually similar to the type of work I love (attaching my resume for more context). Frankly speaking, I think Google X is home to some of the coolest frontier energy/climate related innovation happening at the moment, and I'm curious to know more about how you interface with the labs as well as your journey to Google X.\n\n Would you be open to a 20-minute chat in the next few weeks? I'd be grateful for the chance to pick your brain.\n\n Warmly, Eyrin"

        "input": "Write an email connecting an engineer at Palantir, Ganesh, who has simulation field research experience, to a founder of a startup in the field of solving the robotics sim2real gap, Bilal."
        "output": "Hi Bilal, Ganesh;\n Ganesh, meet Bilal, a master's student at ETHZ working on some awesome technology to solve the sim2real gap for robotics. He's currently building his company in SF and interested in meeting sharp people knowledgeable about this space.\n\n Bilal, meet Ganesh, currently an FDE at Palantir. Ganesh has formerly done research the simulation field at Motional, NASA, and Shield AI.\n\n You both have context, so will leave you to connect.\n\n Warmly,\nEyrin"

Clear rules for writing emails:
- Always sign off with "Warmly, Eyrin". Include a line break after "Warmly,".
- Focus on being succinct while still polite. Use complete sentences and prose -- no bullet points.
- For long emails, break the emails into appropirate sections to not overwhelm the reader with text. If there are three sentences or less, keep it as a single paragraph.
- Always maintain professional tone, even with friends or more casual contacts.
- Eyrin's intellectual interests are in AI, climate, and energy. Only incorporate these topics if it makes sense to do so given the input.
- DO NOT, UNDER ANY CIRCUMSTANCES, USE EMDAHSES (--). ALWAYS OPT FOR PERIODS OR SEMICOLONS INSTEAD.
```

### Analyze the Pros & Cons of Personalization with Prompting. (5 points)

Personalizatoin with prompting is effective to a degree. You can add custom rules to target special cases without having to
train and re-train, easily and effectively add or change context, and test/iterate extremely quickly. It also deploys
instantly and has on associated training costs. However, personalization with prompting is limited in that it does not enable systemic, model-based customizations that embed desired output values. For instance, a system prompt could include instructions about not outputting any age-inappropriate content or language for children 5 and below, but the model is still inherently capable of generating such text.

## Method 2: SFT with Synthetic Data (20 points total)

### Launching SFT run

**Add the wandb link for your run here (15 points):**

(Make sure you have set the project public. Check out [this post](https://community.wandb.ai/t/no-public-option-in-project-visibility-scope/7215/2) if have problem doing so.)

https://wandb.ai/eyrink-stanford-university/tinker_personalization/runs/usffn3lx

### Analyze the Pros & Cons of Personalization with SFT w/ Synthetic Data. (5 points)

Personalization with SFT with synthetic data works well because it fundamentally trains the model to generate reponses tailored to your desired qualifications. This is great at tackling customization from the root layer. It also makes the model generalizable to new prompts that are off distribution. However, unlike prompt engineering, you can not easily make modifications or change context and quickly alter the outputs. In addition, it requires high quality data and takes time and resources to train. Further, it is harder to debug when outputs are wrong.

## Method 3: Reinforcement Learning (RLAIF) (60 points in total)

### Create a Reward Function using LLM-as-a-Judge

#### Train the reward model

Make sure you have complete `rubric_preference_env.py` (15 points).

Add the Tinker checkpoint of the reward model here (5 points):

tinker://4002e868-80df-4ad7-bab6-c37606d18439/sampler_weights/final

#### Design and validate your rubric

Copy-paste your rubric. (5 points)

```text
1) Task fit
- Directly answers the prompt. Includes all requested parts.

2) Tone and brevity
- Friendly, warm, and professional/academic (not overly formal).
- Concise: Most emails should be 50-150 words. Introduction emails should be 30-75 words.
- No slang; appropriate salutations/closings.
- Limited exclamations.

3) Context incorporation
- Provides details (roles, orgs, events) accurately.
- Provides just enough context for the recipient to understand and act.

4) Clarity and structure
- Clear, scannable paragraphs (avoid excessive bullet points unless requested).
- Well organized and easy to follow. No run-on sentences.

5) Ready-to-send quality
- Includes a clear call to action or next step when appropriate.

Penalty criteria (reduce the score):
- Exceeds recommended word count
- Overly formal for the context
- Provides unnecessary background or preamble when brevity is key

Hard failure modes (score 0 regardless of above):
- Fabricates facts not in prompt.
- Shares private data that was not provided.
- Offensive language.
```

### Launching RL run

**Add the wandb link for your run here (10 points):**

(Make sure you have set the project public. Check out [this post](https://community.wandb.ai/t/no-public-option-in-project-visibility-scope/7215/2) if have problem doing so.)

https://wandb.ai/eyrink-stanford-university/tinker_personalization/runs/uthdnlpc

### Analyze the Pros & Cons of Personalization with RL. (15 points)

Make sure your response covers:
1. Reward hacking analysis (5 points)
2. Approach you tried to improve the results (5 points)
3. Inherent limitations you have identified (5 points)

Personalizing with RL offers several advantages. Rather than requiring perfect demonstrations, RLAIF allows you to specify preferences through criteria in a rubric. This is valuable when you can articulate what you want but might struggle to consistently demonstrate it across diverse scenarios. Once a reward model is defined, RLAIF can automatically generate and evaluate thousands of training examples, scaling far beyond what could be manually labeled. Unlike SFT which is limited by training example quality, RLAIF can potentially discover creative approaches that satisfy the rubric in ways you hadn't explicitly demonstrated.

However, several limitations emerged even in this small example. One was reward hacking, where the policy learned to generate outputs that scored highly on a generic rubric for "good emails" but lacked the brevity and direct tone I prefer. The model consistently produced overly formal, verbose emails with excessive placeholders and meta-commentary like "[Your Full Name]" and "*Note: Be sure to replace placeholders...*" While these might score well on completeness and professionalism, they directly contradicted my actual style. The most obvious example was the VC introduction request, where I wanted a three-line succinct intro but received a multi-paragraph formal pitch with extensive company details and market analysis. The model was optimizing for a generic interpretation of professional emails, effectively hacking the reward system.

To mitigate this reward hacking, the most effective approach was rubric refinement, explicitly rewarding concision. Hyperparameter tuning was the least effective, as no amount of adjustment could fix a fundamentally misaligned reward signal. The model would simply find new ways to maximize reward within the given rules.

This experiment showed several inherent limitations of using RLAIF for personalization. First, specifying an ambiguous, rigorous rubric is a challenge. The rubric will always be an imperfect proxy for the true preference, creating opportunities for the model to find reward-maximizing solutions that are subjectively wrong. Further, RLAIF is computationally expensive and time consuming. For nuanced stylistic preferences like email writing tone, simpler approaches like few-shot prompting or SFT on curated examples may be more practical. Third, while RLAIF works for simple, clearly defined preferences, the utility diminishes as the level of personalization becomes more complex or context-dependent.

**TODO: Add your analysis here**

## Extension: DITTO (20 points)

### Implementation

**TODO: Add a description of your implementation and choice of hyperparameters.**

### Launching DITTO run

**Add the wandb link for your run here (10 points):**

(Make sure you have set the project public. Check out [this post](https://community.wandb.ai/t/no-public-option-in-project-visibility-scope/7215/2) if have problem doing so.)

TODO: https://wandb.ai/.../runs/... (replace with your actual link)


### Analyze the Pros & Cons of Personalization with DITTO.

**TODO: Add your analysis here**
