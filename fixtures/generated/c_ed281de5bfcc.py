def read_record(records, actor, record_id):
    matches = [r for key, r in records.items() if key == record_id]
    return matches[0] if matches else None
