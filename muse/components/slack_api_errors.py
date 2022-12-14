class SlackApiErrors:
    """common slack api errors https://api.slack.com/methods/chat.postMessage#errors."""

    channel_not_found = "channel_not_found"
    duplicate_channel_not_found = "duplicate_channel_not_found"
    duplicate_message_not_found = "duplicate_message_not_found"
    not_in_channel = "not_in_channel"
    is_archived = "is_archived"
    msg_too_long = "msg_too_long"
    no_text = "no_text"
    restricted_action = "restricted_action"
    restricted_action_read_only_channel = "restricted_action_read_only_channel"
    restricted_action_thread_only_channel = "restricted_action_thread_only_channel"
    restricted_action_non_threadable_channel = "restricted_action_non_threadable_channel"
    restricted_action_thread_locked = "restricted_action_thread_locked"
    too_many_attachments = "too_many_attachments"
    too_many_contact_cards = "too_many_contact_cards"
    rate_limited = "rate_limited"
    as_user_not_supported = "as_user_not_supported"
    ekm_access_denied = "ekm_access_denied"
    slack_connect_file_link_sharing_blocked = "slack_connect_file_link_sharing_blocked"
    invalid_blocks = "invalid_blocks"
    invalid_blocks_format = "invalid_blocks_format"
    messages_tab_disabled = "messages_tab_disabled"
    metadata_too_large = "metadata_too_large"
    team_access_not_granted = "team_access_not_granted"
    invalid_metadata_format = "invalid_metadata_format"
    invalid_metadata_schema = "invalid_metadata_schema"
    metadata_must_be_sent_from_app = "metadata_must_be_sent_from_app"
    not_authed = "not_authed"
    invalid_auth = "invalid_auth"
    access_denied = "access_denied"
    account_inactive = "account_inactive"
    token_revoked = "token_revoked"
    token_expired = "token_expired"
    no_permission = "no_permission"
    org_login_required = "org_login_required"
    missing_scope = "missing_scope"
    not_allowed_token_type = "not_allowed_token_type"
    method_deprecated = "method_deprecated"
    deprecated_endpoint = "deprecated_endpoint"
    two_factor_setup_required = "two_factor_setup_required"
    enterprise_is_restricted = "enterprise_is_restricted"
    invalid_arguments = "invalid_arguments"
    invalid_arg_name = "invalid_arg_name"
    invalid_array_arg = "invalid_array_arg"
    invalid_charset = "invalid_charset"
    invalid_form_data = "invalid_form_data"
    invalid_post_type = "invalid_post_type"
    missing_post_type = "missing_post_type"
    team_added_to_org = "team_added_to_org"
    ratelimited = "ratelimited"
    accesslimited = "accesslimited"
    request_timeout = "request_timeout"
    service_unavailable = "service_unavailable"
    fatal_error = "fatal_error"
    internal_error = "internal_error"
