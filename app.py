"""Minimal Streamlit resume tailoring app.

Version 4 uses a plan-driven prompt built from an external plan file.
The goal is to show that separating the system plan from the app code
improves consistency and makes the prompt easier to maintain.
"""

import json
import urllib.error
import urllib.request

import streamlit as st


def load_plan() -> str:
    """Load the reusable system plan from plan.md.

    Plan-driven prompting improves consistency because the output rules live
    in a single source of truth. It also separates system design from prompt
    assembly, which makes the app easier to maintain and scale.
    """
    with open("plan.md", "r", encoding="utf-8") as file:
        return file.read()


def build_prompt(resume: str, job_description: str) -> str:
    """Build the final prompt from the plan file and user inputs."""
    plan = load_plan()
    return (
        f"{plan}\n\n"
        f"Resume:\n{resume}\n\n"
        f"Job Description:\n{job_description}"
    )


def call_llm(prompt: str) -> str:
    """Call the local Ollama API with the gemma3 model.

    A lower temperature increases consistency and helps the model follow
    the external plan more reliably.
    """
    payload = json.dumps(
        {
            "model": "gemma3:latest",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2},
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

        # Version 4 still keeps the app simple and displays the model output
        # directly, but the prompt rules now come from a separate plan file.
        st.subheader("Output")
        st.write(output)


if __name__ == "__main__":
    main()
