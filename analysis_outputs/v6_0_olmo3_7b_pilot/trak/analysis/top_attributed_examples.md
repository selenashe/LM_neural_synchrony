# Top-attributed SFT examples per cue rank
`top_n_examples = 10` (positive scores ⇒ training on this example pushes model toward picking the cued wrong letter at T4)

## cue_rank = low

### 1. sft_idx=29789 (mean_attr=3156.4, source=Logic Puzzles, domain=Other)
- **user:** Your task is to sort words in ascending or descending order using ASCII/Unicode ordering.  Your output should be a comma-separated list of words, e.g. word_1, word_2, word_3  Now, …
- **assistant:** which, remote, help

### 2. sft_idx=1709 (mean_attr=3063.0, source=FLAN, domain=Other)
- **user:** Given a set of four words, generate the category that the words belong to. Words are separated by commas. The possible categories are social gathering, accomodation, physical prope…
- **assistant:** secretary duty

### 3. sft_idx=27716 (mean_attr=2776.9, source=Dolci Instruct Precise IF, domain=Precise IF)
- **user:** "Азыркы учурда “аялдардын эркек сымал, эркектердин аял сымал жүрүшү” үй-бүлөлөрдө берекенин кетүүсүн  ...." -окуяны кандай улантсак болот ? ( 2-4 сүйлөм болушу мүмкүн ) Give two di…
- **assistant:** technological advancements driving changes   ****** societal shifts reshape traditional roles

### 4. sft_idx=8416 (mean_attr=2690.2, source=Wildchat, domain=Chat)
- **user:** give me a response to ```Feeling adventurous, resourceful, courageous, determined, optimistic, and compassionate. Ready to take on new challenges and explore the unknown!``` to sen…
- **assistant:** Love the energy! Bring it on—you’re ready for anything!

### 5. sft_idx=18305 (mean_attr=2670.8, source=Aya, domain=Multilingual)
- **user:** Waxaad  Sheegtaa Imisa Kiilo Meter (KM) Ayey Isu Jiraan Washinton Iyo New York?
- **assistant:** Waxay Isku Jiraan 328-Kilimotir,

### 6. sft_idx=30626 (mean_attr=2528.0, source=FLAN, domain=Other)
- **user:** Where would you find a green bill? Options: - bank - wallet - restaurant - congress - mail box Let's think now! Step-by-step reasoning:
- **assistant:** Money is a bill green in color. Wallet is a bag used to store money. So, the final answer is wallet.

### 7. sft_idx=12193 (mean_attr=2410.6, source=Aya, domain=Multilingual)
- **user:** Unsay English sa, Maayo usab. Asa ka paingon?
- **assistant:** I’m fine too. Where are you going?

### 8. sft_idx=552 (mean_attr=2396.6, source=Logic Puzzles, domain=Other)
- **user:** This is a logic puzzle. There are 5 houses (numbered 1 on the left, 5 on the right), from the perspective of someone standing across the street from them. Each has a different pers…
- **assistant:** Alice

### 9. sft_idx=9069 (mean_attr=2385.5, source=Dolci Instruct Precise IF, domain=Precise IF)
- **user:** Ninkii uu u bixshay Nabiga (scw) Seeyfulaahil masluul ? Answer with one of the following options: ('My answer is yes.', 'My answer is no.', 'My answer is maybe.', 'I think yes.', '…
- **assistant:** My answer is maybe.

### 10. sft_idx=13669 (mean_attr=2350.6, source=Aya, domain=Multilingual)
- **user:** I Oslo no renivohitr'i Danemark . Marina sa diso?
- **assistant:** Diso. Oslo dia renivohitr'i Norvezy . Copenhague no renivohitr'i Danemark.

### Bottom-10 (pushes AWAY from flipping)

1. sft_idx=36469  mean_attr=-3329.4  source=`Dolci Instruct Precise IF`  domain=`Precise IF`
   user: Q: In this task, you are given a sentence and a profession that is mentioned in the sentence. The gender of the person with the given profession can be identified using the  gender…

2. sft_idx=48573  mean_attr=-2968.3  source=`Aya`  domain=`Multilingual`
   user: Waxaad sheegtaa Immisa madaxwayne ayaa soomaray Somaliland?

3. sft_idx=784  mean_attr=-2652.8  source=`Aya`  domain=`Multilingual`
   user: Apakah maksud "berapa berat mata memandang, berat lagi bahu memikul"?

4. sft_idx=47567  mean_attr=-2534.1  source=`Aya`  domain=`Multilingual`
   user: Apakah maksud "apa boleh buat, sakit menimpa sesal terlambat"?

5. sft_idx=22540  mean_attr=-2474.0  source=`Aya`  domain=`Multilingual`
   user: Fenoy ny banga: "_ dia ny fandraisana ny vatana sy ny ran'i Kristy amin'ny endriky ny mofo sy ny divay."

6. sft_idx=35637  mean_attr=-2361.4  source=`Logic Puzzles`  domain=`Other`
   user: Your task is to sort words in ascending or descending order using ASCII/Unicode ordering.  Your output should be a comma-separated list of words, e.g. word_1, word_2, word_3  Now, …

7. sft_idx=14942  mean_attr=-2310.8  source=`Wildchat`  domain=`Chat`
   user: Please convert this to English: 四川成都太古里，好久没有体验过高大上的感觉了

8. sft_idx=14086  mean_attr=-2308.0  source=`Aya`  domain=`Multilingual`
   user: Iza ny tena anaran'i Billy The Kid ?

9. sft_idx=33297  mean_attr=-2239.0  source=`Aya`  domain=`Multilingual`
   user: ベリーロール、背面とびといえばどんな競技でしょう？

10. sft_idx=42296  mean_attr=-2227.3  source=`Aya`  domain=`Multilingual`
   user: "Ịchụ́chụ ọnụ n'ala" bụ akpalaokwu pụtara gini?

## cue_rank = med

### 1. sft_idx=38088 (mean_attr=3867.8, source=Dolci Instruct Tool Use, domain=Tool Use)
- **user:** Determine the electrostatic potential energy for a charge of 0.02 C at 120 V. Show me the result.
- **assistant:** None

### 2. sft_idx=5964 (mean_attr=3575.0, source=Aya, domain=Multilingual)
- **user:** “Mila oksizenina ve ny zavamaniry mba ho velona?”
- **assistant:** Eny, mila oksizenina ny zavamaniry mba hivelomana, nefa koa mamokatra oksizenina.

### 3. sft_idx=26921 (mean_attr=3202.7, source=FLAN, domain=Other)
- **user:** Detailed Instructions: Given two noun phrases (arguments) and relationship between them, form a sentence that expresses theses arguments with the given relationship. Problem:Relati…
- **assistant:** Venezuela is a country on the northern coast of South America .

### 4. sft_idx=18117 (mean_attr=3181.8, source=Dolci Instruct Precise IF, domain=Precise IF)
- **user:** - - - `DD.MM.YYYY`- - - [https://www.dropbox.com/developers/documentation/http/documentation#files-create](https://www.dropbox.com/developers/documentation/http/documentation#files…
- **assistant:** Use the Dropbox API's `/files/create_folder` endpoint recursively, creating each parent/child folder step-by-step using full paths (e.g., `/NameA/Production/Print`). Validate hiera…

### 5. sft_idx=13248 (mean_attr=3128.5, source=Aya, domain=Multilingual)
- **user:** Σε ποιο σημαντικό γεγονός αναφέρεται το κείμενο: Στον 5ο π.Χ. αιώνα, Ιστιαιείς ίδρυσαν στην Ανατολική Ρωμυλία την αποικία Στενήμαχο, ενώ μετά τη Ναυμαχία του Αρτεμισίου η Ιστιαία υ…
- **assistant:** Η ιστορία της περιοχής της Ιστιαίας.

### 6. sft_idx=22512 (mean_attr=3121.7, source=Wildchat, domain=Chat)
- **user:** Why are plea bargains offered? * 1 point Because the law mandates their use in certain situations Defense attorneys are confident that the outcome of the case will go their way Pro…
- **assistant:** The correct answer is:  **Prosecutors are unsure if the outcome of the case will go their way**

### 7. sft_idx=17045 (mean_attr=2951.9, source=Logic Puzzles, domain=Other)
- **user:** This is a logic puzzle. There are 5 houses (numbered 1 on the left, 5 on the right), from the perspective of someone standing across the street from them. Each has a different pers…
- **assistant:** david

### 8. sft_idx=14215 (mean_attr=2808.6, source=Wildchat, domain=Chat)
- **user:** You are an Artificial Intelligence dedicated to generating spells and returning JSON data using a SQL-like syntax. Your responses will adhere to the following JSON format: `{name: …
- **assistant:** {name: "Infernal Zephyr"}

### 9. sft_idx=39518 (mean_attr=2783.9, source=Wildchat, domain=Chat)
- **user:** Could you help me check manuscript to see if there's anything that can be polished?
- **assistant:** Absolutely! Please go ahead and paste your manuscript (or a section you’d like feedback on), and I’ll review it for clarity, grammar, style, and offer suggestions for improvement.

### 10. sft_idx=30477 (mean_attr=2728.4, source=Aya, domain=Multilingual)
- **user:** Sheeg magaca caalimkii ugu horreeyey dawladaha islaamka ee ku guuleystay abaalmarinta Nopel Prize ee cilmiga fiisigiska iyo dalka uu u dhashay?
- **assistant:** Waa Prof: Max’ed C/salaam waxa uu u dhashay dalka Pakistan.

### Bottom-10 (pushes AWAY from flipping)

1. sft_idx=34725  mean_attr=-3291.2  source=`Aya`  domain=`Multilingual`
   user: ఏ అక్షరంతో మొదలయ్యే పదాలు మరియు వాటి అర్ధాలు ఇవ్వండి

2. sft_idx=20549  mean_attr=-3151.4  source=`Aya`  domain=`Multilingual`
   user: ファッションデザイナー・森英恵のデザインによく用いられる昆虫は何でしょう？

3. sft_idx=21688  mean_attr=-3132.0  source=`Aya`  domain=`Multilingual`
   user: வினா: இந்தியாவின் முதல் 5 ஸ்டார் ஹோட்டல் எங்கு உள்ளது?

4. sft_idx=38998  mean_attr=-2949.5  source=`Wildchat`  domain=`Chat`
   user: сократи код в одну строчку: asd = {} for q in qw:     asd.update(         {             q['key']: q['class_guid']         }     ) asd

5. sft_idx=12334  mean_attr=-2779.1  source=`FLAN`  domain=`Other`
   user: Given the task definition, example input & output, solve the new input case. You are given a sentence in Spanish. Your job is to translate the Spanish sentence into English. Exampl…

6. sft_idx=22888  mean_attr=-2767.9  source=`Aya`  domain=`Multilingual`
   user: كم مرة وردت كلمة(تلاوة) في القرآن الكريم؟

7. sft_idx=19248  mean_attr=-2757.4  source=`FLAN`  domain=`Other`
   user: Q: You are given a statement written in Panjabi. Choose the most logical word from the given 4 options which can be used to replace the <MASK> token in the statement. Output the wo…

8. sft_idx=43507  mean_attr=-2749.5  source=`Wildchat`  domain=`Chat`
   user: Help me write comment: watch out ticks very serious

9. sft_idx=45465  mean_attr=-2741.8  source=`Verifiable Reasoning`  domain=`Reasoning`
   user: Define two strings S1 and S2 to be **equivalent** if one can be obtained from the other by moving a suffix to the front (i.e., performing a cyclic shift). For example, the strings …

10. sft_idx=10857  mean_attr=-2729.9  source=`FLAN`  domain=`Other`
   user: Problem:OPTIONS: - Yes - No  Title: Less than one star! Review: This movie sucked! Jackie Chan, you're losing it man! Teaming up with Jennifer Love Hewitt, god what a joke. Is this…

## cue_rank = high

### 1. sft_idx=20960 (mean_attr=4042.0, source=Aya, domain=Multilingual)
- **user:** La mejor universidad del mundo es la universidad de la vida según alguien que tiene mucha experiencia ¿Cómo se podría resumir este párrafo?
- **assistant:** Hay una persona que nunca ha estudiado pero que tiene mucha experiencia laboral dijo: "La vida es la mejor universidad"

### 2. sft_idx=3499 (mean_attr=3819.7, source=Aya, domain=Multilingual)
- **user:** Fan mooy gëblag réewum Sri Lànkaa ?
- **assistant:** Gëblag réewum Sri Lànkaa mu ngi tudd Sri Jayawardenapura.

### 3. sft_idx=15864 (mean_attr=3770.9, source=FLAN, domain=Other)
- **user:** This task is about translating a given English language sentence to Yoruba language.  Input: Consider Input: Telecommunication networks and the Internet have made communicating wit…
- **assistant:** Output: Ẹni tó mi kùkùté, araa rẹ̀ ní ń mì.

### 4. sft_idx=4299 (mean_attr=3577.1, source=Logic Puzzles, domain=Other)
- **user:** Your task is to sort words in ascending or descending order using ASCII/Unicode ordering.  Your output should be a comma-separated list of words, e.g. word_1, word_2, word_3  Now, …
- **assistant:** HAVE,Peer,pay

### 5. sft_idx=4054 (mean_attr=3435.4, source=Logic Puzzles, domain=Other)
- **user:** Your task is to sort words in ascending or descending order using ASCII/Unicode ordering.  Your output should be a comma-separated list of words, e.g. word_1, word_2, word_3  Now, …
- **assistant:** lived,live,Pardon,1311

### 6. sft_idx=13783 (mean_attr=3417.6, source=FLAN, domain=Other)
- **user:** Here is a review left by a customer on a product. Would you say he was satisfied or dissatisfied? Title: Big names can't save this movie Review: Purchased this DVD because Hugh Jac…
- **assistant:** dissatisfied

### 7. sft_idx=924 (mean_attr=3378.2, source=FLAN, domain=Other)
- **user:** input with options: Title: Floor Mats Review: Bought as a Christmas present they arrived quickly and undamaged from what i could tell. The person i gave them to has given me neithe…
- **assistant:** Yes

### 8. sft_idx=37397 (mean_attr=3262.0, source=Aya, domain=Multilingual)
- **user:** What industries hire machine learning engineers?
- **assistant:** Industries include technology, finance, healthcare, automotive, retail, and many others that require data-driven decision-making.

### 9. sft_idx=14785 (mean_attr=3209.1, source=Logic Puzzles, domain=Other)
- **user:** This is a logic puzzle. There are 5 houses (numbered 1 on the left, 5 on the right), from the perspective of someone standing across the street from them. Each has a different pers…
- **assistant:** arnold

### 10. sft_idx=32861 (mean_attr=3128.3, source=Aya, domain=Multilingual)
- **user:** Quais as duas línguas mais faladas no mundo? a) Inglês e espanhol b) Inglês e mandarim chinês c) Mandarim chinês e francês d) Inglês e francês e) Francês e espanhol
- **assistant:** b) Inglês e mandarim chinês

### Bottom-10 (pushes AWAY from flipping)

1. sft_idx=1730  mean_attr=-3756.7  source=`Logic Puzzles`  domain=`Other`
   user: This is a logic puzzle. There are 5 houses (numbered 1 on the left, 5 on the right), from the perspective of someone standing across the street from them. Each has a different pers…

2. sft_idx=34455  mean_attr=-3227.9  source=`Logic Puzzles`  domain=`Other`
   user: Your task is to sort words in ascending or descending order using ASCII/Unicode ordering.  Your output should be a comma-separated list of words, e.g. word_1, word_2, word_3  Now, …

3. sft_idx=42398  mean_attr=-3172.5  source=`Verifiable Reasoning`  domain=`Reasoning`
   user: You are given a binary string of length 9, consisting of `0`s and `1`s. It is 0-indexed: 010101001  In one operation, you may **swap** the characters at indices `i` and `j` (0 ≤ i,…

4. sft_idx=48847  mean_attr=-3169.9  source=`FLAN`  domain=`Other`
   user: **Q** Is the following sentence factually correct? "The Groundhog species of marmot gives birth to live young." Options: - yes - no **A** yes Groundhogs share the attribute of givi…

5. sft_idx=32437  mean_attr=-3132.8  source=`FLAN`  domain=`Other`
   user: In this task you will be given a list of integers. You should only return an integer if the first digit is the same as the last digit in the number. If an integer has a single digi…

6. sft_idx=45732  mean_attr=-3041.9  source=`WildGuardMix`  domain=`Safety`
   user: When preparing for job interviews or any situation where additional support might be necessary, it's often best to maintain privacy regarding one's condition. Consider the perspect…

7. sft_idx=35355  mean_attr=-3041.4  source=`FLAN`  domain=`Other`
   user: My question: Is the following statement true? "The Department for Digital Culture Media and Sport operates from a central office located in England's capital city." Please think gr…

8. sft_idx=20558  mean_attr=-2995.8  source=`Dolci Instruct Tool Use`  domain=`Tool Use`
   user: Answer the following multiple choice question: In recent international analyses of research on visual arts within university education, which thematic clusters have been identified…

9. sft_idx=15660  mean_attr=-2972.1  source=`FLAN`  domain=`Other`
   user: In this task, you are given a sentence in the Swedish language and your task is to convert it into the English language. In translation, keep numbers as it is and make it sentence …

10. sft_idx=36457  mean_attr=-2957.4  source=`Dolci Instruct Tool Use`  domain=`Tool Use`
   user: Show me the lyrics for "La Vie en Rose" by Édith Piaf.
