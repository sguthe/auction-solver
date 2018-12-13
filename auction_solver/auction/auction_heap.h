#pragma once

template <class KC, class VC, class K, class V>
void heap_upheap(KC &key, VC &value, int k, K v_key, V v_value)
{
	int j = (k - 1) >> 1;
	//while ((k > 0) && (key[j] <= v_key))
	while ((k > 0) && ((key[j] < v_key) || ((key[j] == v_key) && (value[j] <= v_value))))
	{
		key[k] = key[j];
		value[k] = value[j];
		k = j;
		j = (k - 1) >> 1;
	}
	key[k] = v_key;
	value[k] = v_value;
}

template <class KC, class VC, class K, class V>
void heap_downheap(KC &key, VC &value, int N, int k, K v_key, V v_value)
{
	int j = (k << 1) + 1;
	while (j < N)
	{
		//if ((j + 1 < N) && (key[j] < key[j + 1])) j++;
		if ((j + 1 < N) && ((key[j] < key[j + 1]) || ((key[j] == key[j + 1]) && (value[j] < value[j + 1])))) j++;
		//if (v_key >= key[j]) break;
		if ((v_key > key[j]) || ((v_key == key[j]) && (v_value >= value[j]))) break;
		key[k] = key[j];
		value[k] = value[j];
		k = j;
		j = (k << 1) + 1;
	}
	key[k] = v_key;
	value[k] = v_value;
}

template <class KC, class VC, class K, class V>
void heap_replace(KC &key, VC &value, int N, K v_key, V v_value)
{
	//if (v_key >= key[0]) return;
	if ((v_key > key[0]) || ((v_key == key[0]) && (v_value >= value[0]))) return;
	heap_downheap(key, value, N, 0, v_key, v_value);
}

template <class KC, class VC, class K, class V>
void heap_insert(KC &key, VC &value, int &N, K v_key, V v_value)
{
	N++;
	heap_upheap(key, value, N - 1, v_key, v_value);
}

template <class KC, class VC, class K, class V>
void heap_pop(KC &key, VC &value, int &N)
{
	N--;
	std::swap(key[0], key[N]);
	std::swap(value[0], value[N]);
	heap_downheap(key, value, N, 0, key[0], value[0]);
}

template <class KC, class VC, class K, class V>
void heap_sort(KC &key, VC &value, int N)
{
	while (N > 1)
	{
		N--;
		std::swap(key[0], key[N]);
		std::swap(value[0], value[N]);
		heap_downheap(key, value, N, 0, key[0], value[0]);
	}
}
