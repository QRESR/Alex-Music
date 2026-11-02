"""Декодер VK audio-ссылок вида ...?extra=...#... (audio_api_unavailable).

Точный порт python273/vk_api/vk_api/audio_url_decoder.py (эталон).
Нужен user_id владельца сессии (операция 'i').
"""
from __future__ import annotations

VK_STR = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMN0PQRSTUVWXYZO123456789+/="


class VkAudioUrlDecodeError(Exception):
    pass


def splice(l: list, a: int, b: int, c) -> tuple[list, list]:
    """JS Array.prototype.splice: удалить b начиная с a, вставить c."""
    return l[:a] + [c] + l[a + b:], l[a:a + b]


def vk_o(string: str) -> str:
    result: list[str] = []
    index2 = 0
    i = 0
    for s in string:
        sym_index = VK_STR.find(s)
        if sym_index != -1:
            if index2 % 4 != 0:
                index2 += 1
                i = (i << 6) + sym_index
                result += [chr(0xFF & i >> (-2 * index2 & 6))]
            else:
                i = sym_index
                index2 += 1
    return "".join(result)


def vk_r(string: str, i: str) -> str:
    vk_str2 = VK_STR + VK_STR
    vk_str2_len = len(vk_str2)
    result: list[str] = []
    for s in string:
        index = vk_str2.find(s)
        if index != -1:
            offset = index - int(i)
            if offset < 0:
                offset += vk_str2_len
            result += [vk_str2[offset]]
        else:
            result += [s]
    return "".join(result)


def vk_xor(string: str, i: str) -> str:
    xor_val = ord(i[0])
    return "".join(chr(ord(s) ^ xor_val) for s in string)


def vk_s_child(t: str, e) -> list[int]:
    i = len(t)
    if not i:
        return []
    o: list[int] = []
    e = int(e)
    for a in range(i - 1, -1, -1):
        e = (i * (a + 1) ^ e + a) % i
        o.append(e)
    return o[::-1]


def vk_s(t: str, e) -> str:
    i = len(t)
    if not i:
        return t
    o = vk_s_child(t, e)
    t_list = list(t)
    for a in range(1, i):
        t_list, y = splice(t_list, o[i - 1 - a], 1, t_list[a])
        t_list[a] = y[0]
    return "".join(t_list)


def vk_i(t: str, e, user_id: int) -> str:
    return vk_s(t, int(e) ^ user_id)


def decode_audio_url(string: str, user_id: int) -> str:
    if "?extra=" not in string:
        return string
    vals = string.split("?extra=", 1)[1].split("#")
    tstr = vk_o(vals[0])
    ops_list = vk_o(vals[1]).split("\x09")[::-1]
    for op_data in ops_list:
        split_op_data = op_data.split("\x0b")
        cmd = split_op_data[0]
        arg = split_op_data[1] if len(split_op_data) > 1 else None
        if cmd == "v":
            tstr = tstr[::-1]
        elif cmd == "r":
            tstr = vk_r(tstr, arg)
        elif cmd == "x":
            tstr = vk_xor(tstr, arg)
        elif cmd == "s":
            tstr = vk_s(tstr, arg)
        elif cmd == "i":
            tstr = vk_i(tstr, arg, user_id)
        else:
            raise VkAudioUrlDecodeError(f'Unknown decode cmd: "{cmd}"')
    return tstr


def decode(raw: str, user_id: int) -> str | None:
    """Безопасная обёртка: прямой URL или None."""
    try:
        if "?extra=" not in raw:
            return raw if raw.startswith("http") else None
        out = decode_audio_url(raw, user_id)
        return out if out.startswith("http") else None
    except Exception:  # noqa: BLE001
        return None
