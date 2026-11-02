# Local transcript data

Do not commit collected transcripts by default. The `speechprefix import-govinfo` command can append eligible CPD records to `speeches.jsonl`. You can also add permitted records manually, one JSON object per line, with these required string fields:

```json
{"speech_id":"unique-id","date":"2024-01-01","event":"event name","source":"URL or archive reference","text":"Transcript text..."}
```

Use ISO dates (`YYYY-MM-DD`) so chronological ordering is reliable. Keep source and license/permission notes. Do not include text you are not allowed to store or process. The data is user-supplied; this repository does not download transcripts.
