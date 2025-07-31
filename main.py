from transformers import AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
import torch
import wandb

from src.reward import custom_reward_fn

model_name = "Qwen/Qwen3-0.6B"

def main():
    # Load a pre-trained causal language model and its tokenizer
    device = torch.device("mps")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)

    # Create a minimal dummy dataset. Each sample must include a "prompt" column.
    train_dataset = [
        {"prompt": "Discuss the impact of social media on modern communication.", "completion": "Social media has significantly changed the way people interact, providing instant communication..."},
        {"prompt": "Explain the causes and effects of climate change.", "completion": "Climate change is driven by both natural and human factors, primarily the emission of greenhouse gases..."},
        {"prompt": "Analyze the role of technology in education.", "completion": "Technology has revolutionized education by providing online resources, interactive learning tools..."},
        {"prompt": "Compare and contrast democracy and authoritarianism.", "completion": "Democracy and authoritarianism are two contrasting political systems, differing in governance structure..."},
        {"prompt": "Respond ONLY using paragraphs. Evaluate the benefits and drawbacks of space exploration.", "completion": "Space exploration has expanded human knowledge, leading to technological advancements..."},
        {"prompt": "Respond ONLY using paragraphs. Discuss the ethical implications of artificial intelligence.", "completion": "Artificial intelligence presents ethical concerns related to privacy, job displacement..."},
        {"prompt": "Respond ONLY using paragraphs. Examine the historical significance of the Industrial Revolution.", "completion": "The Industrial Revolution marked a turning point in human history, leading to rapid industrialization..."},
        {"prompt": "Respond ONLY using paragraphs. How does globalization affect local cultures?", "completion": "Globalization has led to cultural exchange, economic growth, but also concerns over cultural homogenization..."},
        {"prompt": "Respond ONLY using paragraphs. What are the key factors influencing economic inequality?", "completion": "Economic inequality stems from disparities in income distribution, education access, and policy decisions..."},
        {"prompt": "Respond ONLY using paragraphs. Discuss the importance of mental health awareness in society.", "completion": "Mental health awareness has become crucial in modern society, influencing policies, workplaces..."},
        {"prompt": "Respond ONLY using paragraphs. Examine the role of women in leadership throughout history.", "completion": "Women have played significant leadership roles, though often facing systemic barriers..."},
        {"prompt": "Respond ONLY using paragraphs. Analyze the effects of urbanization on the environment.", "completion": "Urbanization leads to economic growth but also contributes to pollution, deforestation..."},
        {"prompt": "Respond ONLY using paragraphs. Discuss the importance of financial literacy for young adults.", "completion": "Financial literacy is crucial for managing debt, savings, and long-term financial stability..."},
        {"prompt": "Respond ONLY using paragraphs. What are the long-term effects of plastic pollution?", "completion": "Plastic pollution affects marine life, ecosystems, and human health through microplastics..."},
        {"prompt": "Respond ONLY using paragraphs. Explain the impact of colonialism on modern geopolitics.", "completion": "Colonialism reshaped borders, economies, and cultures, leaving lasting impacts on global relations..."},
        {"prompt": "Respond ONLY using paragraphs. Evaluate the pros and cons of remote work in the post-pandemic era.", "completion": "Remote work offers flexibility and cost savings, but also challenges in productivity and collaboration..."},
        {"prompt": "Respond ONLY using paragraphs. How does literature reflect the values of its time?", "completion": "Literature often mirrors societal values, serving as a reflection of historical and cultural contexts..."},
        {"prompt": "Respond ONLY using paragraphs. What are the major causes of the Great Depression?", "completion": "The Great Depression was triggered by stock market crashes, bank failures, and poor economic policies..."},
        {"prompt": "Respond ONLY using paragraphs. Analyze the significance of the Renaissance in shaping modern thought.", "completion": "The Renaissance led to advancements in science, art, and philosophy, influencing modern perspectives..."},
        {"prompt": "Respond ONLY using paragraphs. Discuss the role of censorship in media and its impact on freedom of speech.", "completion": "Censorship in media raises debates over balancing security, misinformation, and free expression..."},
        {"prompt": "Respond ONLY using paragraphs. Explore the impact of gentrification on urban communities.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Discuss the role of social media in activism and social justice movements.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Analyze the portrayal of mental health in popular media.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Examine the influence of celebrity culture on societal values.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Discuss the challenges faced by immigrants in integrating into new cultures.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Evaluate the effectiveness of affirmative action policies in promoting diversity.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Explore the concept of cultural appropriation versus cultural appreciation.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Discuss the role of art in addressing social issues.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Analyze the impact of fast fashion on the environment and labor practices.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Examine the significance of indigenous languages in preserving cultural heritage.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Discuss the potential of blockchain technology beyond cryptocurrency.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Analyze the ethical considerations of genetic engineering.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Explore the future of renewable energy sources.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Discuss the impact of automation on the job market.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Evaluate the benefits and risks of 5G technology.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Analyze the role of big data in shaping modern business strategies.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Explore the implications of quantum computing for cybersecurity.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Discuss the potential of virtual reality in education and training.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Examine the ethical dilemmas surrounding surveillance technology.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Analyze the impact of social media algorithms on user behavior.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Discuss the importance of biodiversity in ecosystems.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Analyze the causes and effects of deforestation.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Explore the concept of sustainable agriculture.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Discuss the impact of climate change on wildlife.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Evaluate the effectiveness of international climate agreements.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Analyze the role of corporations in environmental sustainability.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Explore the benefits and challenges of electric vehicles.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Discuss the importance of water conservation in urban areas.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Examine the impact of tourism on natural environments.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Analyze the potential of carbon capture and storage technologies.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Discuss the benefits of mindfulness and meditation.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Analyze the impact of sleep deprivation on mental health.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Explore the role of nutrition in preventing chronic diseases.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Discuss the importance of physical activity in maintaining health.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Evaluate the effectiveness of alternative medicine practices.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Analyze the impact of stress on the immune system.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Explore the concept of holistic health and wellness.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Discuss the challenges of healthcare accessibility in rural areas.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Examine the role of community support in mental health recovery.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Analyze the impact of telemedicine on healthcare delivery.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Discuss the benefits of project-based learning in education.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Analyze the impact of standardized testing on student performance.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Explore the role of extracurricular activities in student development.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Discuss the importance of early childhood education.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Evaluate the effectiveness of online learning platforms.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Analyze the challenges faced by special education teachers.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Explore the concept of lifelong learning and its importance.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Discuss the role of mentorship in professional development.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Examine the impact of multilingual education on cognitive development.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Analyze the benefits and drawbacks of homeschooling.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Discuss the significance of the Cold War in shaping global politics.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Analyze the impact of the French Revolution on modern democracy.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Explore the role of women in the Civil Rights Movement.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Discuss the causes and consequences of the Vietnam War.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Evaluate the effectiveness of the United Nations in maintaining global peace.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Analyze the impact of the Arab Spring on Middle Eastern politics.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Explore the significance of the Silk Road in ancient trade.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Discuss the role of propaganda in World War II.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Examine the legacy of colonialism in Africa.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Analyze the impact of the Industrial Revolution on labor practices.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Discuss the themes of love and loss in Shakespeare's plays.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Analyze the use of symbolism in F. Scott Fitzgerald's 'The Great Gatsby'.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Explore the role of magic realism in Latin American literature.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Discuss the influence of Greek mythology on modern storytelling.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Evaluate the impact of the Harlem Renaissance on African American literature.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Analyze the portrayal of women in Jane Austen's novels.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Explore the concept of the 'unreliable narrator' in literature.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Discuss the significance of the Beat Generation in American literature.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Examine the use of satire in Jonathan Swift's 'Gulliver's Travels'.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Analyze the themes of identity and belonging in Toni Morrison's works.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Discuss the concept of existentialism and its relevance today.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Analyze the ethical implications of utilitarianism.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Explore the philosophy of stoicism and its application in modern life.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Discuss the debate between free will and determinism.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Evaluate the moral arguments for and against vegetarianism.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Analyze the concept of virtue ethics in Aristotle's philosophy.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Explore the role of empathy in moral decision-making.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Discuss the ethical considerations of human cloning.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Examine the philosophy of transcendentalism and its influence on American thought.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Analyze the concept of the 'social contract' in political philosophy.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Discuss the significance of the discovery of DNA.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Analyze the impact of the theory of relativity on modern physics.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Explore the potential of space tourism.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Discuss the role of the Hubble Space Telescope in astronomy.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Evaluate the benefits and risks of genetic modification in agriculture.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Analyze the impact of the Human Genome Project on medical research.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Explore the concept of dark matter and dark energy in the universe.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Discuss the significance of the discovery of exoplanets.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Examine the role of the Large Hadron Collider in particle physics.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Analyze the potential of nanotechnology in medicine.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Discuss the impact of globalization on international trade.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Analyze the causes and effects of income inequality.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Explore the role of entrepreneurship in economic development.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Discuss the significance of the gig economy in modern employment.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Evaluate the effectiveness of microfinance in alleviating poverty.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Analyze the impact of corporate social responsibility on business practices.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Explore the concept of the sharing economy and its implications.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Discuss the role of venture capital in startup funding.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Examine the impact of economic sanctions on international relations.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Analyze the benefits and drawbacks of free trade agreements.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Write a short story about a time traveler who gets stuck in the past.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Create a poem that explores the theme of loneliness.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Write a letter to your future self about your current aspirations and fears.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Describe a memorable childhood experience and its impact on you.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Write a dialogue between two historical figures who never met.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Create a fictional world and describe its unique features.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Write a personal essay about overcoming a significant challenge.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Describe a dream you had and analyze its possible meanings.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Write a short story about a person who discovers they have a superpower.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Create a poem that celebrates the beauty of nature.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Discuss the impact of podcasts on modern storytelling.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Analyze the role of humor in coping with stress.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Explore the significance of rituals and traditions in different cultures.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Discuss the benefits and challenges of living in a multigenerational household.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Evaluate the effectiveness of community gardens in urban areas.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Analyze the impact of pet ownership on mental health.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Explore the concept of 'digital detox' and its benefits.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Discuss the role of hobbies in personal development.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Examine the significance of public art in urban spaces.", "completion": ""},
        {"prompt": "Respond ONLY using paragraphs. Analyze the impact of volunteering on community well-being.", "completion": ""}
    ]

    wandb.init(
      project="grpo-training",
      config={
        "model": model_name,
        "learning_rate": 3e-4,
        "max_steps": 100,
        "scheduler": "constant_with_warmup",
        "max_grad_norm": 0.2
      }
    )

    # Define a GRPO configuration. Adjust parameters as needed.
    config = GRPOConfig(
        learning_rate=3e-6,
        output_dir="output",
        max_steps=100,
        importance_sampling_level="sequence",
        lr_scheduler_type="constant_with_warmup",
        report_to=["wandb"],
        logging_steps=1,
        max_grad_norm=0.2,
        beta=0.04,
        fp16=True
        #max_completion_length=1024
    )

    # Initialize the GRPOTrainer with the model, training dataset and custom reward function.
    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        reward_funcs=custom_reward_fn,
    )

    # Start training
    trainer.train()

    wandb.finish()

if __name__ == "__main__":
    main()
