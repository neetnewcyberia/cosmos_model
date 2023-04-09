import json
import logging

logger = logging.getLogger(__name__)

def run(user_input, inference_fn) -> None:
    generation_settings = {
        "do_sample": True,
        "max_new_tokens": 196,
        "temperature": 0.5,
        "top_p": 0.9,
        "top_k": 0,
        "typical_p": 1.0,
        "repetition_penalty": 1.05,
        "penalty_alpha": 0.6
    }
    history_for_model = gr.State([])
    
    def _run_inference(
            model_history,
            user_input,
            generation_settings,
            *char_setting_states,
        ):
            '''
            Runs inference on the model, and formats the returned response for
            the Gradio state and chatbot component.
            '''
            char_name = char_setting_states[0]
            user_name = char_setting_states[1]

            # If user input is blank, format it as if user was silent
            if user_input is None or user_input.strip() == "":
                user_input = "..."

            inference_result = inference_fn(model_history, user_input,
                                            generation_settings,
                                            *char_setting_states)

            inference_result_for_gradio = inference_result \
                .replace(f"{char_name}:", f"**{char_name}:**") \
                .replace("<USER>", user_name) \
                .replace("\n", "<br>") # Gradio chatbot component can display br tag as linebreak

            model_history.append(f"You: {user_input}")
            model_history.append(inference_result)

            return model_history

    def _regenerate(
        model_history,
        generation_settings,
        *char_setting_states,
    ):
        '''Regenerates the last response.'''
        return _run_inference(
            model_history[:-2],
            model_history[-2].replace("You: ", ""),
            generation_settings,
            *char_setting_states,
        )

    def _save_chat_history(model_history, *char_setting_states):
        '''Saves the current chat history to a .json file.'''
        char_name = char_setting_states[0]
        with open(f"{char_name}_conversation.json", "w") as f:
            f.write(json.dumps({"chat": model_history}))
        return f"{char_name}_conversation.json"

    def _load_chat_history(file_obj, *char_setting_states):
        '''Loads up a chat history from a .json file.'''
        # #############################################################################################
        # TODO(TG): Automatically detect and convert any CAI dump files loaded in to Pygmalion format #
        # #############################################################################################

        # https://stackoverflow.com/questions/5389507/iterating-over-every-two-elements-in-a-list
        def pairwise(iterable):
            # "s -> (s0, s1), (s2, s3), (s4, s5), ..."
            a = iter(iterable)
            return zip(a, a)

        char_name = char_setting_states[0]
        user_name = char_setting_states[1]

        file_data = json.loads(file_obj.decode('utf-8'))
        model_history = file_data["chat"]

        return model_history

    return _run_inference(
        model_history,
        user_input,
        generation_settings,
        *char_setting_states,
    )
