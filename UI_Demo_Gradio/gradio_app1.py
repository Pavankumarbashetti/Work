import gradio as gr


def framework_selection(frameworks):
    return f"You selected: {', '.join(frameworks)}"


interface = gr.Interface(
    fn=framework_selection,
    inputs=gr.CheckboxGroup(
        choices=["Langchain", "Langgraph", "crewAI","LLamaIndex"], label="Select Frameworks"
    ),
    outputs="text",
    title="Framework Selection",
    description="Select your preferred frameworks.",
)

interface.launch(share=True)
