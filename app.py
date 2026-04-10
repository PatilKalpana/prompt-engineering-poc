"""Minimal Streamlit resume tailoring app.

Version 2 keeps the same simple app structure, but uses a better prompt.
The goal is to show that stronger prompt design can improve output quality
without adding extra parsing, validation, or backend complexity.
"""

import json
import urllib.error
import urllib.request

import streamlit as st


def build_prompt(resume: str, job_description: str) -> str:
    """Build a structured prompt that gives the model a clear job and format.

    This prompt is better than Version 1 because it defines:
    - the model's role
    - the exact task
    - explicit constraints
    - a strict output structure

    That reduces engineering effort because we rely on the prompt itself
    instead of adding post-processing code to clean messy outputs.
    """
    return (
        "You are an expert career coach and resume strategist.\n\n"
        "Your task is to tailor the candidate's resume to match the job "
        "description as closely as possible while staying truthful to the "
        "candidate's actual experience.\n\n"
        "Constraints:\n"
        "- Do not include any extra explanations, commentary, rationale, or notes.\n"
        "- Do not ask follow-up questions.\n"
        "- Do not add information that is not supported by the resume.\n"
        "- Keep the response concise and job-focused.\n\n"
        "Output Format:\n"
        "Tailored Resume:\n"
        "<tailored resume here>\n\n"
        "Match Score:\n"
        "<number from 0 to 100>\n\n"
        "Missing Skills:\n"
        "- <skill 1>\n"
        "- <skill 2>\n\n"
        "Suggestions:\n"
        "- <suggestion 1>\n"
        "- <suggestion 2>\n\n"
        f"Resume:\n{resume}\n\n"
        f"Job Description:\n{job_description}"
    )


def call_llm(prompt: str) -> str:
    """Call the local Ollama API with the gemma3 model.

    A slightly lower temperature makes the output more stable and reduces
    unnecessary creativity, which helps the prompt produce cleaner structure.
    """
    payload = json.dumps(
        {
            "model": "gemma3:latest",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.4},
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

        # Version 2 still keeps the app simple and displays the model output
        # directly, but the prompt now asks the model to include score,
        # missing skills, and suggestions in the response itself.
        st.subheader("Output")
        st.write(output)


if __name__ == "__main__":
    main()
