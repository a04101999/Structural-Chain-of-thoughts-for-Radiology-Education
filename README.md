# Structural-Chain-of-thoughts-for-Radiology-Education

Radiology education requires trainees to develop both perceptual and inter-
pretive expertise, yet the feedback required to develop these skills remain
scarce due to the demanding schedules of experienced radiologists. This
lack of personalized guidance makes it difficult for learners to understand
not just what errors they made, but also the reason why those errors oc-
curred and how to refine their reasoning skills. Although Large Language
Models (LLMs) and Large Multimodal Models (LMMs) have shown promise
in radiology applications, they struggle with fine-grained multimodal rea-
soning. Specifically, these models struggle in detecting subtle cross-modal
patterns, such as variations in gaze behavior and diagnostic decisions. These
small yet critical differences in how experts and novices allocate visual at-
tention can reveal underlying perceptual gaps, which are often overlooked
by current AI-driven approaches. To address these limitations, we introduce
Structural Chain of Thoughts (SCoT)—a novel framework that enhances
AI sensitivity to nuanced multimodal differences by structuring gaze data
and diagnostic reasoning into a thought graph. By leveraging a structural
prior, SCoT systematically identifies key perceptual and interpretive discrep-
ancies, allowing models to provide targeted, context-aware feedback. This
structured approach not only highlights missed findings but also explains
the reasoning behind perceptual errors, turning them into learning opportu-
nities. Applied within radiology education, SCoT bridges the gap between
expert and novice performance, offering a scalable solution for AI-driven di-
agnostic training. We further contribute a simulated dataset of perceptual
errors, facilitating future research into multimodal reasoning and educational
AI in medical imaging

# Simulated Error Dataset:

Due to unavailability of Real world error dataset, this study was conducted on the simulated error dataset:

https://drive.google.com/drive/folders/1RzlGzvJ9Dl01dgrNlhNedY7JWhQ3pmJE?usp=sharing

It contains two files:
1) Error dataset with missing or masked fixations
2) Labels for the the cases ( whethere missed or not )
