import heapq


class PRIORITY:
    def __init__(self, l):
        self.content = l
        heapq.heapify(self.content)

    def __len__(self):
        return len(self.content)

    def get(self):
        return heapq.heappop(self.content)

    def push(self, item):
        heapq.heappush(self.content, item)

    def get_max(self):
        return heapq.nlargest(1, self.content)


class LIST:
    def __init__(self, l):
        self.content = list(l)

    def __len__(self):
        return len(self.content)

    def get(self):
        return self.content.pop(0)

    def push(self, item):
        self.content.append(item)

    def get_max(self):
        vals = [e[0] for e in self.content]
        return max(vals)
