# ci: run
"""A 60-line re-implementation of the Agent / Task / Crew pattern (sequential process).

It mirrors the vocabulary of CrewAI (role, goal, backstory, description,
expected_output, context) but is NOT CrewAI code. The 'llm' is any function
prompt -> text; here a recorder stub is used so that the exact prompt each agent
receives can be inspected and measured.
"""
from dataclasses import dataclass, field


@dataclass
class Agent:
    role: str
    goal: str
    backstory: str
    llm: callable = None

    def system(self, **inputs):
        fmt = lambda s: s.format(**inputs)
        return (f"You are {self.role}. {fmt(self.backstory)}\n"
                f"Your personal goal is: {fmt(self.goal)}")


@dataclass
class Task:
    description: str
    expected_output: str
    agent: Agent
    context: list = field(default_factory=list)   # tasks whose outputs are shown to this one
    output: str = None


class Crew:
    def __init__(self, agents, tasks, verbose=False):
        self.agents, self.tasks, self.verbose = agents, tasks, verbose
        self.prompts = []

    def kickoff(self, inputs):
        prev = None
        for t in self.tasks:
            ctx_tasks = t.context or ([prev] if prev else [])   # default: previous output
            ctx = "\n\n".join(c.output for c in ctx_tasks)
            prompt = t.agent.system(**inputs) + "\n\nCurrent task: " + t.description.format(**inputs)
            prompt += "\nThis is the expected criteria for your final answer: " + t.expected_output
            if ctx:
                prompt += "\n\nThis is the context you are working with:\n" + ctx
            self.prompts.append(prompt)
            t.output = t.agent.llm(prompt)
            prev = t
        return self.tasks[-1].output


def words(s):
    return len(s.split())


def stub_llm(tag, n_words):
    """Returns a fixed-length text, so prompt sizes are comparable."""
    return lambda prompt: f"[{tag}] " + " ".join(["lorem"] * n_words)


if __name__ == "__main__":
    planner = Agent("Content Planner", "Plan engaging and factually accurate content on {topic}",
                    "You are planning a blog article about the topic: {topic}.", stub_llm("plan", 300))
    writer = Agent("Content Writer", "Write an insightful opinion piece about {topic}",
                   "You base your writing on the work of the Content Planner.", stub_llm("draft", 900))
    editor = Agent("Editor", "Edit a blog post to align with the organization's style.",
                   "You receive a blog post from the Content Writer and review it.", stub_llm("final", 900))
    plan = Task("1. Prioritize the latest trends on {topic}.\n2. Develop an outline.",
                "A content plan with an outline.", planner)
    write = Task("Use the plan to write a blog post on {topic}.", "A blog post in markdown.", writer)
    edit = Task("Proofread the given blog post.", "A polished blog post in markdown.", editor)

    crew = Crew([planner, writer, editor], [plan, write, edit])
    out = crew.kickoff({"topic": "Future of AI"})
    sizes = [words(p) for p in crew.prompts]
    gen = [300, 900, 900]
    print("prompt words per stage:", sizes)
    print("generated words per stage:", gen)
    print("total input words:", sum(sizes), " total output words:", sum(gen))
    # the writer's prompt contains the plan, the editor's prompt contains the draft
    assert "[plan]" in crew.prompts[1] and "[draft]" in crew.prompts[2] and "[plan]" not in crew.prompts[2]
    # the final result is the LAST task's output (the notebook displayed the middle one)
    assert out.startswith("[final]") and write.output.startswith("[draft]")
    print("last prompt, first 300 chars:\n", crew.prompts[2][:300])
    print("crew_mini self-test passed")
