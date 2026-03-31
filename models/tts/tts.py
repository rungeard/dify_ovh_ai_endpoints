from typing import Any, Mapping, Optional

from dify_plugin.entities.model import AIModelEntity, I18nObject

from dify_plugin.interfaces.model.openai_compatible.tts import OAICompatText2SpeechModel
from models.ovh_credentials import build_ovh_credentials


class OpenAIText2SpeechModel(OAICompatText2SpeechModel):
    def _invoke(
        self,
        model: str,
        tenant_id: str,
        credentials: dict,
        content_text: str,
        voice: str,
        user: Optional[str] = None,
    ) -> Any:
        normalized_credentials = build_ovh_credentials(credentials)
        return super()._invoke(model, tenant_id, normalized_credentials, content_text, voice, user)

    def validate_credentials(self, model: str, credentials: dict, user: Optional[str] = None) -> None:
        normalized_credentials = build_ovh_credentials(credentials)
        return super().validate_credentials(model, normalized_credentials, user)

    def get_customizable_model_schema(
        self, model: str, credentials: Mapping | dict
    ) -> AIModelEntity:
        credentials = build_ovh_credentials(credentials)
        entity = super().get_customizable_model_schema(model, credentials)

        if "display_name" in credentials and credentials["display_name"] != "":
            entity.label = I18nObject(en_US=credentials["display_name"])

        return entity
