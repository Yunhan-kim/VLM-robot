import json
from pathlib import Path

from utils.llm_util import LLMBase
from utils.io_util import load_txt

class VLMPlanner(LLMBase):
    """LLM TAMP Planner"""

    def __init__(
        self,
        planner_prompt_file: str,
        env_desc: str,        
    ):
        
        LLMBase.__init__(self)

        # load planning prompt template
        prompt_template_folder = Path(__file__).resolve().parent.parent / "prompts"
        planning_prompt_template = load_txt(
            prompt_template_folder / f"{planner_prompt_file}.txt"
        )

        # load problem file
        domain_desc = self._domain_desc(env_desc)
        self._planning_prompt = planning_prompt_template.replace("{domain_desc}", domain_desc)        
    
    def _domain_desc(self, env_desc: str):
        domain_desc = ''
        for i in range(len(env_desc)):
            target_obj = env_desc[i][0]
            target_obb = env_desc[i][3]
            domain_desc += '   - **{}**: Position {}, size {} x {} (x, y)\n'.format(target_obj, str(list(target_obb[0][0])), str(target_obb[0][1][0]), str(target_obb[0][1][1]))
        return domain_desc            


    def plan(self, question: str):
        # plan
        planning_prompt = self._planning_prompt        
        planning_prompt = planning_prompt.replace("{question}", question)
        
        plan, reasoning = None, None

        llm_output = self.prompt_llm(planning_prompt)                
        llm_output = json.loads(llm_output.strip('```json\n').strip('```'))

        plan = llm_output['Full Plan']
        reasoning = llm_output["Reasoning"]

        return plan, reasoning