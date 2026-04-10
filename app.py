"""Minimal Streamlit resume tailoring app.

Version 1 is intentionally naive:
- simple prompt construction
- simple LLM call
- raw output display
- placeholder score
"""

import json
import urllib.error
import urllib.request

import streamlit as st


def build_prompt(resume: str, job_description: str) -> str:
    """Build the simplest possible prompt for resume tailoring."""
    return (
        "Tailor this resume for this job:\n\n"
        f"Resume:\n{resume}\n\n"
        f"Job Description:\n{job_description}"
    )


def call_llm(prompt: str) -> str:
    """Call the local Ollama API with the gemma3 model."""
    payload = json.dumps(
        {
            "model": "gemma3:latest",
            "prompt": prompt,
            "stream": False,
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

        # Show the raw LLM output without extra formatting or processing.
        st.subheader("Raw Output")
        st.text(output)

        # Version 1 uses a placeholder score only.
        st.subheader("Score")
        st.write("N/A")


if __name__ == "__main__":
    main()
