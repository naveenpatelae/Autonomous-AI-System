#!/usr/bin/env python3
# =====================================================================
# 🤖 META-AGENT FACTORY (Fixing the Swarm)
#
# Replaces the generic EphemeralAgent with a deterministic micro-swarm.
# Spawns ephemeral agents for:
#   1. Routing (Tool Selection)
#   2. Extraction (JSON Parameter generation)
#   3. Validation (Schema enforcement & hallucination correction)
# =====================================================================

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("MetaAgentFactory")

# ─────────────────────────────────────────────────────────────────────
# TOOL REGISTRY & SCHEMAS
# ─────────────────────────────────────────────────────────────────────
@dataclass
class ToolSchema:
    name: str
    description: str
    required_params: List[str]
    param_types: Dict[str, str]

class ToolRegistry:
    """Holds the definitions of tools the swarm can physically actuate."""
    def __init__(self):
        self._tools: Dict[str, ToolSchema] = {}

    def register(self, tool: ToolSchema):
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[ToolSchema]:
        return self._tools.get(name)

    def get_all_descriptions(self) -> str:
        desc = []
        for t in self._tools.values():
            desc.append(f"- {t.name}: {t.description} (Requires: {', '.join(t.required_params)})")
        return "\n".join(desc)


# ─────────────────────────────────────────────────────────────────────
# THE META-AGENT FACTORY
# ─────────────────────────────────────────────────────────────────────
class MetaAgentFactory:
    """
    Dynamically spawns specialized micro-agents to safely execute tasks.
    Uses an LLM to Route -> Extract -> Validate.
    """
    
    MAX_RETRIES = 3

    def __init__(self, llm_fn: Callable[[str], str], registry: ToolRegistry):
        self._llm = llm_fn
        self._registry = registry

    def _spawn_router(self, task_description: str) -> str:
        """Micro-agent #1: Selects the correct tool."""
        prompt = (
            f"You are a Tool Routing Agent.\n"
            f"TASK: {task_description}\n\n"
            f"AVAILABLE TOOLS:\n{self._registry.get_all_descriptions()}\n\n"
            f"Which tool should be used? Return ONLY the exact tool name. "
            f"If no tool fits, return 'NONE'."
        )
        try:
            response = self._llm(prompt).strip().replace("'", "").replace('"', '')
            return response
        except Exception as e:
            logger.error(f"[RouterAgent] Error: {e}")
            return "NONE"

    def _spawn_extractor(self, task_description: str, tool: ToolSchema, feedback: str = "") -> dict:
        """Micro-agent #2: Extracts JSON parameters."""
        schema_format = {k: f"<{v}>" for k, v in tool.param_types.items()}
        
        prompt = (
            f"You are a Parameter Extraction Agent.\n"
            f"TASK: {task_description}\n"
            f"TOOL: {tool.name}\n"
            f"REQUIRED FORMAT: {json.dumps(schema_format)}\n\n"
        )
        if feedback:
            prompt += f"PREVIOUS ERROR: {feedback}\nFix the JSON to resolve this error.\n\n"
            
        prompt += "Return ONLY a valid JSON object matching the required format."
        
        try:
            raw_response = self._llm(prompt)
            # Extract JSON block
            raw_json = re.sub(r'```json|```', '', raw_response).strip()
            return json.loads(raw_json)
        except json.JSONDecodeError as e:
            logger.warning(f"[ExtractorAgent] Invalid JSON generated: {e}")
            return {}
        except Exception as e:
            logger.error(f"[ExtractorAgent] LLM Error: {e}")
            return {}

    def _spawn_validator(self, args: dict, tool: ToolSchema) -> Tuple[bool, str]:
        """Micro-agent #3: Strict schema validation (Programmatic)."""
        if not args:
            return False, "Empty or invalid JSON provided."
            
        missing = [p for p in tool.required_params if p not in args or args[p] in ("", "UNKNOWN", None)]
        if missing:
            return False, f"Missing required parameters: {', '.join(missing)}"
            
        # Optional: Add LLM-based semantic validation here if needed
        return True, "Valid"

    def execute_task(self, task_id: str, task_description: str) -> dict:
        """
        The main factory pipeline: Route -> Extract -> Validate (Loop) -> Return.
        """
        logger.info(f"🏭 [MetaFactory] Processing Task {task_id}: {task_description[:50]}...")
        t0 = time.time()
        
        # 1. Routing
        tool_name = self._spawn_router(task_description)
        tool = self._registry.get_tool(tool_name)
        
        if not tool:
            logger.warning(f"🏭 [MetaFactory] Router selected invalid tool: {tool_name}")
            return {"status": "failed", "reason": f"No valid tool found (Router picked '{tool_name}')"}
            
        logger.info(f"🏭 [MetaFactory] Router selected tool: {tool.name}")
        
        # 2. Extraction & Validation Loop
        feedback = ""
        final_args = {}
        success = False
        
        for attempt in range(self.MAX_RETRIES):
            logger.debug(f"🏭 [MetaFactory] Extractor attempt {attempt + 1}/{self.MAX_RETRIES}")
            args = self._spawn_extractor(task_description, tool, feedback)
            
            is_valid, msg = self._spawn_validator(args, tool)
            if is_valid:
                final_args = args
                success = True
                break
            else:
                feedback = msg
                logger.warning(f"🏭 [MetaFactory] Validator rejected args: {msg}")
                
        elapsed = round((time.time() - t0) * 1000, 1)
        
        if success:
            logger.info(f"🏭 [MetaFactory] Task {task_id} successfully parsed in {elapsed}ms.")
            return {
                "status": "success",
                "tool": tool.name,
                "arguments": final_args,
                "elapsed_ms": elapsed
            }
        else:
            logger.error(f"🏭 [MetaFactory] Task {task_id} failed after {self.MAX_RETRIES} attempts.")
            return {
                "status": "failed",
                "reason": f"Validator rejected generated arguments: {feedback}",
                "elapsed_ms": elapsed
            }

# ─────────────────────────────────────────────────────────────────────
# INTEGRATION TEST SUITE
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    print("🏭 Meta-Agent Factory Self-Test\n")

    # 1. Setup Mock Registry
    registry = ToolRegistry()
    registry.register(ToolSchema(
        name="create_event",
        description="Creates a calendar event",
        required_params=["title", "time"],
        param_types={"title": "string", "time": "ISO-8601 datetime string", "location": "string optional"}
    ))
    registry.register(ToolSchema(
        name="send_email",
        description="Sends an email to a contact",
        required_params=["recipient", "subject", "body"],
        param_types={"recipient": "email address", "subject": "string", "body": "string"}
    ))

    # 2. Setup Mock LLM
    def mock_llm(prompt: str) -> str:
        # Simulate Router
        if "Tool Routing Agent" in prompt:
            if "yoga class" in prompt.lower():
                return "create_event"
            elif "email" in prompt.lower():
                return "send_email"
            return "NONE"
            
        # Simulate Extractor
        if "Parameter Extraction Agent" in prompt:
            if "MISSING_TIME" in prompt: # Simulate fixing a validation error
                return '{"title": "Yoga class at Isha", "time": "2026-04-07T07:00:00", "location": "Isha"}'
            if "yoga class" in prompt.lower() and "PREVIOUS ERROR" not in prompt:
                # Deliberately fail the first attempt to test the Validator loop
                return '{"title": "Yoga class at Isha", "location": "Isha"}' 
        return "{}"

    # 3. Run Factory
    factory = MetaAgentFactory(llm_fn=mock_llm, registry=registry)
    
    print("=== Test 1: Full Pipeline with Validation Retry ===")
    result = factory.execute_task("task_001", "Schedule a yoga class at Isha for 2026-04-07 at 7:00 AM")
    
    assert result["status"] == "success", f"Pipeline failed: {result}"
    assert result["tool"] == "create_event", "Wrong tool routed"
    assert "time" in result["arguments"], "Extractor failed to fix missing parameter"
    
    print(f"  ✅ Task Processed: {result['tool']} with args {result['arguments']}")
    print(f"  ✅ Elapsed Time: {result['elapsed_ms']}ms\n")
    
    print("=== Test 2: Unrelated Task ===")
    result2 = factory.execute_task("task_002", "Tell me a joke")
    assert result2["status"] == "failed", "Factory should have failed on unrelated task"
    print(f"  ✅ Factory correctly rejected unrelated task.\n")
    
    print("✅ All Meta-Agent Factory tests passed.")