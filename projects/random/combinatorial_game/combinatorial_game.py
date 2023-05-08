import numpy as np
import random
import colorama
from colorama import Fore

# dostępne tryby gry:
# 1. losowy: komputer wybiera pozycję całkowicie losowo,
# 2. defensywny: komputer chroni się przed przegraną - blokuje możliwe ruchy wygrywające gracza,
# 3. ofensywny: komputer zależy na ułożeniu jak najszybciej ciągu wygrywającego,
# 4. zoptymalizowany: połączenie poprzednich dwóch w stosunku 1:1,
# 5. zmodyfikowany: połączenie strategii def. i ofen. w stosunku α:β wybranym przez gracza.


def find_arytm_sqnc_all(idx, k):
    result = []
    for i in range(len(idx)):
        for j in range(len(idx)):
            r = abs(idx[i] - idx[j])
            curr = 0
            index = i
            while index < len(idx):
                if idx[index] == idx[i] + curr * r:
                    curr += 1
                if curr == k:
                    result.append((idx[i],r))
                index += 1
    return result

def find_arytm_sqnc(idx, k):
    for i in range(len(idx)):
        for j in range(len(idx)):
            r = abs(idx[i] - idx[j])
            curr = 0
            index = i
            while index < len(idx):
                if idx[index] == idx[i] + curr * r:
                    curr += 1
                if curr == k:
                    return True
                index += 1
    return False

def check_if_win(Board, k):
    idx1 = []
    idx2 = []
    for i in range(len(Board)):
        if Board[i][1] == 1:
            idx1.append(Board[i][0])
        if Board[i][1] == 2:
            idx2.append(Board[i][0])

    if find_arytm_sqnc(idx1, k):
        print(Fore.BLUE + "Gracz", end=""); print(Fore.RESET + " wygrał!")
        result = find_arytm_sqnc_all(idx1, k)
        print(f"Wygrywający ciąg to: ", end="")
        for i in range(k):
            print(f"{result[0][0] + i * result[0][1]}", end=" ")
        return False
    if find_arytm_sqnc(idx2, k):
        print(Fore.RED + "Komputer", end=""); print(Fore.RESET + " wygrał!")
        result = find_arytm_sqnc_all(idx2, k)
        print(f"Wygrywający ciąg to: ", end="")
        for i in range(k):
            print(f"{result[0][0] + i * result[0][1]}", end=" ")
        return False

    cnt = 0
    for i in range(len(Board)):
        if Board[i][1] == -1:
            cnt += 1
    if cnt == 0:
        print("Remis.")
        return False

    return True

def player_turn(Board):
    print("Wybiera ", end=""); print(Fore.BLUE + 'Gracz', end=""); print(Fore.RESET + '. ', end="")
    index = input("Podaj liczbę do pokolorowania: ")

    while True:
        if index.isnumeric() and int(index) <= len(Board):
            break
        else:
            print("Niepoprawna wartość! Spróbuj jeszcze raz.")
            index = input("Podaj liczbę do pokolorowania: ")

    index = int(index)

    while Board[index - 1][1] == 1 or Board[index - 1][1] == 2:
        index = input("Liczba jest już pokolorowana, podaj inną: ")
        while True:
            if index.isnumeric() and int(index) <= len(Board):
                break
            else:
                print("Niepoprawna wartość! Spróbuj jeszcze raz.")
                index = input("Podaj liczbę do pokolorowania: ")
        index = int(index)

    Board[index - 1][1] = 1


def pc_random_turn(Board):
    index = random.randint(0, len(Board) - 1)
    while Board[index][1] != -1:
        index = random.randint(0, len(Board) - 1)
    Board[index][1] = 2

    print(Fore.RED + "Komputer", end=""), print(Fore.RESET + f" wybiera pozycję {index+1}.")


def pc_opt_turn(Board, k, alpha, beta):
    n = len(Board)
    lst = list(range(1, n + 1))

    all_sqnc = find_arytm_sqnc_all(lst, k)

    unique = []
    for sqnc in all_sqnc:
        if sqnc not in unique:
            unique.append(sqnc)

    idx1 = []
    idx2 = []
    for i in range(n):
        if Board[i][1] == 1:
            idx1.append(Board[i][0])
        if Board[i][1] == 2:
            idx2.append(Board[i][0])

    score_def = []
    score_off = []
    for sqnc in unique:
        tmp = 0
        mix = False
        for j in range(k):
            if sqnc[0] + j * sqnc[1] in idx1:
                tmp += beta
                mix = True
        for j in range(k):
            if sqnc[0] + j * sqnc[1] in idx2:
                tmp = 0
                if mix:
                    tmp = -1
        score_def.append(tmp)

    for sqnc in unique:
        tmp = 0
        mix = False
        for j in range(k):
            if sqnc[0] + j * sqnc[1] in idx2:
                tmp += alpha
                mix = True
        for j in range(k):
            if sqnc[0] + j * sqnc[1] in idx1:
                tmp = 0
                if mix:
                    tmp = -1
        score_off.append(tmp)

    sqncs_score = np.column_stack((unique, score_def, score_off))

    for i in range(len(sqncs_score)):
        if sqncs_score[i][3] == k - 1:
            for j in range(k):
                if Board[sqncs_score[i][0] + j * sqncs_score[i][1] - 1][1] == -1:
                    print(Fore.RED + "Komputer", end=""), print(Fore.RESET + f" wybiera pozycję {Board[sqncs_score[i][0] + j * sqncs_score[i][1] - 1][0]}.")
                    Board[sqncs_score[i][0] + j * sqncs_score[i][1] - 1][1] = 2
                    return
    for i in range(len(sqncs_score)):
        if sqncs_score[i][2] == k - 1:
            for j in range(k):
                if Board[sqncs_score[i][0] + j * sqncs_score[i][1] - 1][1] == -1:
                    print(Fore.RED + "Komputer", end=""), print(Fore.RESET + f" wybiera pozycję {Board[sqncs_score[i][0] + j * sqncs_score[i][1] - 1][0]}.")
                    Board[sqncs_score[i][0] + j * sqncs_score[i][1] - 1][1] = 2
                    return

    best_sqncs = []
    best_score_sqnc = -1
    for i in range(len(sqncs_score)):
        if sqncs_score[i][2] + sqncs_score[i][3] == best_score_sqnc:
            best_sqncs.append((sqncs_score[i][0], sqncs_score[i][1]))
        if sqncs_score[i][2] + sqncs_score[i][3] > best_score_sqnc:
            best_score_sqnc = sqncs_score[i][2] + sqncs_score[i][3]
            best_sqncs.clear()
            best_sqncs.append((sqncs_score[i][0], sqncs_score[i][1]))

    lst1 = list(range(1, n + 1))
    lst2 = [0] * n

    moves_score = np.column_stack((lst1, lst2))

    for sqnc in best_sqncs:
        for j in range(k):
            moves_score[sqnc[0]+j*sqnc[1]-1][1] += 1

    best_score_move = -1
    best_moves = []
    for i in range(len(moves_score)):
        if moves_score[i][1] > best_score_move and Board[i][1] == -1:
            best_score_move = moves_score[i][1]
            best_moves.clear()
            best_moves.append(i)
        if moves_score[i][1] == best_score_move and Board[i][1] == -1:
            best_moves.append(i)

    index = random.randint(0, len(best_moves) - 1)
    while Board[best_moves[index]][1] != -1:
        index = random.randint(0, len(best_moves) - 1)
    Board[best_moves[index]][1] = 2

    print(Fore.RED + "Komputer", end=""), print(Fore.RESET + f" wybiera pozycję {Board[best_moves[index]][0]}.")

def game():
    n, k, m, p, alpha, beta = preproc()
    lst1 = list(range(1, n + 1))
    lst2 = [-1] * n
    Board = np.column_stack((lst1, lst2))

    cnt = 0
    while check_if_win(Board, k):
        if cnt % 2 == (m - 1):
            player_turn(Board)
        else:
            if p == 1: # tryb losowy
                pc_random_turn(Board)
            elif p == 2: # tryb defensywny
                pc_opt_turn(Board, k, 0, 1)
            elif p == 3:  # tryb ofensywny
                pc_opt_turn(Board, k, 1, 0)
            elif p == 4:  # tryb zbalansowany
                pc_opt_turn(Board, k, 1, 1)
            elif p == 5: # tryb zmodyfikowany
                pc_opt_turn(Board, k, alpha, beta)

        print_board(Board)
        print("\n------")

        cnt += 1

def print_board(Board):
    n = len(Board)
    for i in range(n-1):
        print(i+1, end=" | ")
    print(f"{n} |")

    for i in range(n):
        if len(str(i+1)) == 1:
            if Board[i][1] == -1:
                print(Fore.RESET + "_",  end=""), print(" | ",  end="")
            if Board[i][1] == 1:
                print(Fore.BLUE + 'G',  end=""), print(Fore.RESET + " | ",  end="")
            if Board[i][1] == 2:
                print(Fore.RED + "K",  end=""), print(Fore.RESET + " | ",  end="")
        elif len(str(i+1)) == 2:
            if Board[i][1] == -1:
                print(Fore.RESET + "_ ",  end=""), print(" | ",  end="")
            if Board[i][1] == 1:
                print(Fore.BLUE + 'G ',  end=""), print(Fore.RESET + " | ",  end="")
            if Board[i][1] == 2:
                print(Fore.RED + 'K ',  end=""), print(Fore.RESET + " | ",  end="")


def preproc():
    n = input("Podaj n: ")
    while True:
        if n.isnumeric():
            n = int(n)
            break
        else:
            print("Podaj poprawną liczbę.")
            n = input("Podaj n: ")

    k = input("Podaj k: ")
    while True:
        if k.isnumeric() and int(n) >= int(k):
            k = int(k)
            break
        else:
            print("Podaj poprawną liczbę.")
            k = input("Podaj k: ")


    print("Wybierz kto zaczyna: 1 - ", end=''); print(Fore.BLUE + 'gracz', end=''); print(Fore.RESET + ', 2 - ', end=''); print(Fore.RED + 'komputer', end='')
    m = input(Fore.RESET + ": ")
    while True:
        if m.isnumeric() and (int(m) in [1,2]):
            m = int(m)
            break
        else:
            print("Podaj poprawną liczbę."); print("Wybierz kto zaczyna: 1 - ", end=''); print(Fore.BLUE + 'gracz', end=''); print(Fore.RESET + ', 2 - ', end=''); print(Fore.RED + 'komputer', end='')
            m = input(Fore.RESET + ": ")

    p = input("Wybierz tryb gry komputera: 1 - losowy, 2 - defensywny, 3 - ofensywny, 4 - zoptymalizowany, 5 -  zmodyfikowany: ")
    while True:
        if p.isnumeric() and (int(p) in [1,2,3,4,5]):
            p = int(p)
            break
        else:
            print("Podaj poprawną liczbę.")
            p = input("Wybierz tryb gry komputera: 1 - losowy, 2 - defensywny, 3 - ofensywny, 4 - zoptymalizowany, 5 -  zmodyfikowany: ")

    alpha = 1
    beta = 1

    if int(p) == 5:
        alpha = input("Podaj jak bardzo ofensywnie ma grać komputer (podstawowo 1): ")
        while True:
            if alpha.isnumeric() and int(alpha) > 0:
                alpha = int(alpha)
                break
            else:
                print("Podaj poprawną liczbę.")
                alpha = input("Podaj jak bardzo ofensywnie ma grać komputer (podstawowo 1): ")

        beta = input("Podaj jak bardzo defensywnie ma grać komputer (podstawowo 1): ")
        while True:
            if beta.isnumeric() and int(beta) > 0:
                beta = int(beta)
                break
            else:
                print("Podaj poprawną liczbę.")
                beta = input("Podaj jak bardzo defensywnie ma grać komputer (podstawowo 1): ")

    return n, k, m, p, alpha, beta


def main():
    game()


if __name__ == '__main__':
    main()
