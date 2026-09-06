
# A Beginner's Guide to AI Movie Recommendations & The "Popularity Trap"
### Understanding How Algorithms Learn Niche vs. Blockbuster Movies (MF vs. PMF vs. BPR)

---

## 1. The Big Picture: Why Did We Do This? 🎯

Have you ever wondered why streaming apps like Netflix, YouTube, or Spotify always seem to recommend the same massive hits (like *Stranger Things* or *The Avengers*) over and over again, while thousands of great indie films remain hidden in the dark?[cite: 3]

This common frustration in artificial intelligence is known as **Popularity Bias**.[cite: 3]

### The Main Goal of This Project
We wanted to peek inside the "brain" of recommendation algorithms to understand:[cite: 3]
1. **Why do AI models ignore hidden gems?**[cite: 3]
2. **How does the mathematical way we train an AI change how it views movies?**[cite: 3]
3. **Can changing the AI's learning style help it understand niche movies just as well as blockbusters?**[cite: 3]

To answer these questions, we took a real-world movie dataset, trained three different AI recommendation models, extracted the "mental maps" (called **embeddings**) the models built, and compared them visually.[cite: 3]

---

## 2. The Data & The "Superstar" Problem 📊

We used the famous **MovieLens 1M** dataset:[cite: 3]
* 👥 **6,040 real users**[cite: 3]
* 🎬 **3,706 movies**[cite: 3]
* ⭐ **1,000,209 ratings** (on a 1 to 5 star scale)[cite: 3]

### The "80/20" Rule on Steroids
When we counted how many times each movie was rated, we found an extreme imbalance:[cite: 3]

| Movie Group | Description | How Many Movies? | How Many Ratings Did They Get? | % of All Ratings |
| :--- | :--- | :--- | :--- | :--- |
| **Short-Tail** 🌟 *(Blockbusters)* | Highly popular ($\ge 250$ ratings) | **1,216 movies** | **808,922 ratings** | **80.9%** |
| **Mid-Tail** ⚖️ *(Regular Movies)* | Moderately popular ($50 - 249$ ratings) | **1,298 movies** | **168,917 ratings** | **16.9%** |
| **Long-Tail** 📉 *(Hidden Gems/Niche)* | Rarely watched ($< 50$ ratings) | **1,192 movies** | **22,370 ratings** | **2.2%** |


```

All 1,000,000 Ratings Visualized:
████████████████████████████████████████ (80.9%)  Short-Tail (Blockbusters)
████████                                 (16.9%)  Mid-Tail (Regular)
█                                        (2.2%)   Long-Tail (Niche Gems)

```

### The Big Insight:
Notice something wild: **The Blockbusters and the Niche Gems have roughly the same number of movies (~1,200 each)**.[cite: 3] 
Yet, the blockbusters hog almost **81%** of all user activity, while the bottom 1,200 movies share a tiny **2.2%**.[cite: 3]

> 💡 **Analogy:** Imagine a classroom of 30 students where the teacher spends 50 minutes talking to just 3 loud kids, and only has 1 minute left for the other 27 kids.[cite: 3] The teacher will get to know those 3 kids deeply, but will know almost nothing about the rest![cite: 3]

---

## 3. What Did We Do & How Did We Do It? 🛠️

We built a 5-step experiment:[cite: 3]

1. **Prepared the Data:** Split the 1 million ratings into an 80% training set (to teach the models) and a 20% test set (to evaluate them).[cite: 3]
2. **Trained 3 Different AI Models:** We tested three popular algorithms:[cite: 3]
   * **MF** (Standard Matrix Factorization)[cite: 3]
   * **PMF** (Probabilistic Matrix Factorization)[cite: 3]
   * **BPR** (Bayesian Personalized Ranking)[cite: 3]
3. **Set the "Clue Limit":** Each model was instructed to describe every user and every movie using a profile of **32 numbers** (called a 32-dimensional embedding).[cite: 3]
4. **Extracted the Embeddings:** We pulled out the final 32-number profile for all 3,675 active movies.[cite: 3]
5. **Drew 2D Maps (Using t-SNE):** Humans cannot visualize 32 dimensions at once.[cite: 3] So, we used a smart visualization tool called **t-SNE** to squash those 32 numbers down onto a flat, 2-dimensional scatter plot—like projecting a globe onto a paper map.[cite: 3]

---

## 4. What on Earth is an "Embedding"? 🧠

Think of an **embedding** as a movie's **taste profile** or a list of **32 hidden ingredients**.[cite: 3] 

Instead of humans writing manual tags like "Comedy" or "Action", the AI invents its own abstract scales through trial and error, such as:[cite: 3]
* *Number 1:* How dark and serious is the tone? (from -1.0 to +1.0)[cite: 3]
* *Number 2:* How fast-paced is the action?[cite: 3]
* *Number 3:* How much romantic drama is involved?[cite: 3]
* *Number 4:* Does it feature mind-bending plot twists?[cite: 3]
* ...all the way up to *Number 32*.[cite: 3]

Likewise, **every user gets their own 32 numbers** representing what ingredients they enjoy.[cite: 3]

### How Recommendations Are Made:
To calculate how much you will like a movie, the computer multiplies your 32 numbers by the movie's 32 numbers (a math trick called a **dot product**):[cite: 3]

$$\text{Predicted Match Score} = (\text{Your 32 Numbers}) \times (\text{Movie's 32 Numbers})$$

* If your numbers and the movie's numbers match up (both high on sci-fi, both low on romance), the score is **huge** 🚀 $\rightarrow$ *Recommended!*[cite: 3]
* If the numbers conflict, or if the movie's numbers are all zeroes, the score is **near zero** 📉 $\rightarrow$ *Never recommended!*[cite: 3]

---

## 5. Meet the 3 Models: Three Different Ways to Learn 🥊

Not all AI models learn the same way.[cite: 3] Here is how our three contenders work in plain English:[cite: 3]


```

+-----------------------------------------------------------------------------+
| 1. Matrix Factorization (MF) - "The Strict Star Grader"                     |
| Focus: "Can I predict the EXACT star rating (1 to 5) for watched movies?"  |
| Strategy: Pointwise (looks at one user and one movie at a time).            |
| Weakness: It ignores movies you HAVEN'T watched. Niche movies rarely update.|
+-----------------------------------------------------------------------------+

+-----------------------------------------------------------------------------+
| 2. Probabilistic Matrix Factorization (PMF) - "The Cautious Grader"         |
| Focus: "Same as MF, but assumes by default that all movies are average."    |
| Strategy: If a movie has very few ratings, the math aggressively pulls its  |
|           numbers straight toward (0, 0, 0... 0) to avoid wild guesses.     |
+-----------------------------------------------------------------------------+

+-----------------------------------------------------------------------------+
| 3. Bayesian Personalized Ranking (BPR) - "The Matchmaker / Comparison Boy" |
| Focus: "I don't care about stars. Did you like Movie A MORE than Movie B?"  |
| Strategy: Pairwise (compares a watched movie against an UNWATCHED movie).   |
| Superpower: It actively learns from unwatched movies!                        |
+-----------------------------------------------------------------------------+

```

---

## 6. What Did the Maps Look Like? (The Plots & Embedding Shapes) 🗺️

When we plotted the 2D maps of all 3,675 movies, we color-coded every dot:[cite: 3]
* 🟠 **Orange dots:** Short-tail (Blockbusters with 250+ ratings)[cite: 3]
* 🟣 **Purple dots:** Mid-tail (Regular movies with 50-249 ratings)[cite: 3]
* 🟢 **Green dots:** Long-tail (Niche movies with < 50 ratings)[cite: 3]

Here is a visual sketch of what each model produced:[cite: 3]


```

```
    MF & PMF MAP                                 BPR MAP

```

(The "Collapsed Ball" Shape)             (The "Universal Spread" Shape)

```
🟠                 🟠                     🟢      🟠         🟣

```

🟠   ┌─────────────┐   🟠                🟠      🟢       🟠       🟢
│  🟢 🟢 🟢   │                            🟣         🟢
🟠   │  🟢 🟢 🟢   │    🟠                  🟠       🟣      🟠
│  🟢 🟢 🟢   │                     🟢      🟠         🟢    🟠
🟠   └─────────────┘   🟠                   🟣         🟢        🟣
🟠                                       🟠

```

### What We Saw:
1. **In MF & PMF (The Collapse):**[cite: 3]
   * Almost all the **green dots** (niche movies) are squished together into a tight, dense ball right in the center.[cite: 3]
   * The **orange dots** (blockbusters) fan out widely into distinct neighborhoods around the outside.[cite: 3]
2. **In BPR (The Great Dispersion):**[cite: 3]
   * The dense green ball in the center **completely disappears**![cite: 3]
   * The green dots are scattered across the whole map, mixing naturally with purple and orange dots in every neighborhood.[cite: 3]

---

## 7. Why Did the Shapes Change? (The "Secret Sauce") 🔍

Why did MF/PMF crush niche movies into a ball, while BPR spread them out?[cite: 3]

### Why MF and PMF Collapsed:
* **The Penalty Rule (Regularization):** In AI training, there is a penalty designed to prevent numbers from blowing up to crazy extremes.[cite: 3] This penalty acts like a rubber band constantly pulling every number toward **zero**.[cite: 3]
* **Starvation:** A niche movie with only 3 ratings only gets pushed away from zero 3 times during the entire training process.[cite: 3]
* **Result:** The rubber band wins![cite: 3] The movie's 32 numbers collapse to $(0.01, 0.00, -0.01 \dots)$, meaning the AI knows nothing about it.[cite: 3] On our map, all these "near-zero" movies end up in the exact same spot: a dense ball in the middle.[cite: 3]

### Why BPR Spread Them Out:
* **BPR Learns from Unwatched Movies:** BPR constantly trains by picking a movie you watched, and comparing it against a movie you **didn't watch**.[cite: 3]
* **The Math of Probability:** You have probably only watched 50 movies out of 3,700.[cite: 3] That means over 3,600 movies sit in your "unwatched" pile![cite: 3]
* Even a niche movie with only 2 ratings is in the unwatched pile of thousands of users.[cite: 3] During training, it gets drawn as the "unwatched movie" hundreds of times![cite: 3]
* **The Directional Push:**[cite: 3]
  * When an Action fan watches *Die Hard*, the unwatched niche movie gets pushed **away** from Action.[cite: 3]
  * When a Romance fan watches *The Notebook*, that same niche movie gets pushed **away** from Romance.[cite: 3]
* Because thousands of users with totally different tastes push that niche movie in different directions, it is forced away from zero and given a distinct location on the map![cite: 3]

---

## 8. What Does This Change Specify in the Real World? 🎬

What happens when a user opens their app under each model?[cite: 3]


```

+--------------------------+--------------------------------------------------+
| Model                    | What the User Actually Experiences               |
+--------------------------+--------------------------------------------------+
| Matrix Factorization     | "The Boring Echo Chamber" 😴                     |
| (MF / PMF)               | Because all niche movies have vectors near zero, |
|                          | their match score is always zero. Only big       |
|                          | blockbusters get recommended. Everyone gets the  |
|                          | same popular movies.                             |
+--------------------------+--------------------------------------------------+
| Bayesian Personalized    | "The Serendipitous Explorer" 🍿                   |
| Ranking (BPR)            | Because niche movies have real positions on the  |
|                          | map, if you love 1980s Japanese anime, a rare    |
|                          | 1980s anime movie can achieve a high match score |
|                          | and be recommended to you!                       |
+--------------------------+--------------------------------------------------+

```

### The Hidden Catch with BPR:
Nothing in engineering is free![cite: 3] BPR assumes that if you didn't watch a movie, you *disliked* it.[cite: 3] But in reality, maybe you just *didn't know it existed*.[cite: 3] 
* **The Risk:** Sometimes BPR pushes a niche movie in the wrong direction simply because people haven't discovered it yet, introducing a bit of random noise.[cite: 3]

---

## 9. Key Takeaways & What's Next for This Project 🚀

### In Summary:
1. **Popularity bias is a real mathematical trap:** 80% of data comes from 30% of items.[cite: 3]
2. **Pointwise models (MF/PMF) fail niche items:** They starve them of updates, collapsing their vectors to zero.[cite: 3]
3. **Pairwise models (BPR) rescue niche items:** By treating unwatched movies as comparisons, they actively carve out meaningful embeddings for rare titles.[cite: 3]

### Great Additions for Your Presentation or Paper:
* **Measure Catalog Coverage:** Calculate what percentage of all 3,700 movies the system actually recommends across all users. (BPR will score much higher than MF!).[cite: 3]
* **Measure Novelty:** Score how surprising or unexpected the recommendations are.[cite: 3]
* **Smart Negative Sampling:** Instead of picking unwatched movies completely at random in BPR, test picking unwatched popular movies more often to teach the AI even faster.[cite: 3]
* **Graph Neural Networks (LightGCN):** Test a modern Graph AI, which connects users and movies like a spiderweb to pass knowledge from popular movies over to niche movies.[cite: 3]

```