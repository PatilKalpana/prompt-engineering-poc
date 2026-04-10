"""Minimal Streamlit resume tailoring app.

Version 3 keeps the same simple app structure, but uses a few-shot prompt.
The goal is to show that a worked example can improve output consistency
without adding extra parsing, validation, or backend complexity.
"""

import json
import urllib.error
import urllib.request

import streamlit as st


def build_prompt(resume: str, job_description: str) -> str:
    """Build a few-shot prompt using one example to demonstrate the pattern.

    Few-shot prompting improves consistency because the model can imitate
    the example structure directly instead of inferring the format from
    abstract instructions alone.

    This reduces engineering effort further than Version 2 because the
    example itself teaches the response style, which lowers the need for
    extra output cleanup or formatting rules in code.
    """
    return (
        "Example\n\n"
        "Job Description:\n"
        "Looking for a backend developer with Python, Django, REST APIs, "
        "PostgreSQL, and Git experience. Knowledge of Docker is a plus.\n\n"
        "Resume:\n"
        "Software Developer with experience in Python, Flask, MySQL, Git, "
        "and web application development. Built internal tools and APIs for "
        "business teams. Familiar with deployment and debugging.\n\n"
        "Output:\n"
        "Tailored Resume:\n"
        "Software Developer with experience in Python, backend web "
        "development, API development, databases, and Git. Built internal "
        "tools and APIs for business teams, with strong hands-on experience "
        "in developing and debugging web applications.\n\n"
        "Match Score:\n"
        "82\n\n"
        "Missing Skills:\n"
        "- Django\n"
        "- PostgreSQL\n"
        "- Docker\n\n"
        "Suggestions:\n"
        "- Emphasize any backend API work that aligns with REST API development.\n"
        "- Highlight database experience in a way that maps more closely to PostgreSQL.\n"
        "- Mention deployment tools or container exposure if relevant.\n\n"
        "Now generate the output for the following input.\n\n"
        f"Job Description:\n{job_description}\n\n"
        f"Resume:\n{resume}\n\n"
        "Output:\n"
    )


def call_llm(prompt: str) -> str:
    """Call the local Ollama API with the gemma3 model.

    A lower temperature makes the output more stable, which helps the model
    follow the example format more consistently.
    """
    payload = json.dumps(
        {
            "model": "gemma3:latest",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3},
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request) as response:
            response_data = json.loads(response.read().decode("utf-8"))
            return response_data.get("response", "")
    except urllib.error.URLError as exc:
        return f"LLM call failed: {exc}"


def main() -> None:
    """Render the Streamlit app and handle the Generate action."""
    st.title("Resume Tailoring Tool")

    # Input area for the original resume text.
    resume = st.text_area("Resume", height=220)

    # Input area for the target job description.
    job_description = st.text_area("Job Description", height=220)

    # Generate the tailored resume output when the button is clicked.
    if st.button("Generate"):
        prompt = build_prompt(resume, job_description)
        output = call_llm(prompt)

        # Version 3 still keeps the app simple and displays the model output
        # directly. The example in the prompt guides the model toward a more
        # consistent response shape without adding backend parsing logic.
        st.subheader("Output")
        st.write(output)


if __name__ == "__main__":
    main()
