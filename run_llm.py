from takuma_llm_toolkit import TextGenerator


def main() -> None:
    generator = TextGenerator()
    model_name = input("Enter the model name: ")
    user_input = "Tell me about the Backpropagation method."
    if len(model_name) == 0:
        model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    generator.run(model_name, user_input)


if __name__ == "__main__":
    main()
