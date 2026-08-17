
### Rego Policy Example (`tool_policy.rego`)
###To check this code locally, ensure your target OPA engine running on `http://localhost:8181` is processing a rule package structure like this:

##rego
package ai.tool_guard

default allow = false

# Allow access if no denial requirements hit
allow if {
    not deny
}

# Example Guardrail: Stop execution if division-by-zero expression patterns emerge
deny if {
    input.tool_name == "calculate"
    contains(input.arguments.expression, "/0")
}


# Rule to permit calculation actions
allow if {
    input.tool_name == "calculate"
}

# Rule to permit current datetime actions
allow if {
    input.tool_name == "get_current_datetime"
}

# Rule to permit week day lookup actions
allow if {
    input.tool_name == "get_day_of_week"
}